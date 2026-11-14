"""Train a transaction-level IBM AML detector on HI-Small_Trans.csv.

This detects the dataset's laundering label, not PaySim/IEEE fraud. Account
identifiers are never ML inputs. Splits use only transaction timestamps.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from experiment_tracking import finish_experiment, start_experiment
from ibm_history_features import CausalAccountHistory, CausalHistoryStream, HISTORY_FEATURES
from train_ieee import file_provenance, metrics, operating_profiles, ranking_metrics, threshold_analysis

EXPECTED_HEADER = [
    'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account',
    'Amount Received', 'Receiving Currency', 'Amount Paid',
    'Payment Currency', 'Payment Format', 'Is Laundering',
]
READ_COLUMNS = ['Timestamp', 'From Bank', 'To Bank', 'Amount Paid',
                'Receiving Currency', 'Payment Currency', 'Payment Format', 'Is Laundering']
HISTORY_READ_COLUMNS = READ_COLUMNS + ['Account', 'Account.1']
NUMERIC = ['amount_paid', 'log_amount_paid', 'same_bank', 'same_currency', 'hour', 'weekday']
CATEGORICAL = ['receiving_currency', 'payment_currency', 'payment_format']
FEATURES = NUMERIC + CATEGORICAL
CHUNK_SIZE = 100_000


def validate_file(path: Path) -> None:
    if path.name != 'HI-Small_Trans.csv' or not path.is_file():
        raise ValueError('Use the IBM AML HI-Small_Trans.csv variant only')
    with path.open('r', newline='', encoding='utf-8-sig') as stream:
        header = next(csv.reader(stream), None)
    if header != EXPECTED_HEADER:
        raise ValueError(f'Unexpected IBM AML header: {header!r}')


def iter_raw(path: Path, include_accounts: bool = False):
    dtypes = {'From Bank': 'string', 'To Bank': 'string', 'Amount Paid': 'float32',
              'Receiving Currency': 'string', 'Payment Currency': 'string',
              'Payment Format': 'string', 'Is Laundering': 'int8'}
    if include_accounts:
        dtypes.update({'Account': 'string', 'Account.1': 'string'})
    yield from pd.read_csv(path, usecols=HISTORY_READ_COLUMNS if include_accounts else READ_COLUMNS, dtype=dtypes,
                           chunksize=CHUNK_SIZE, low_memory=False)


def parse_time(values: pd.Series) -> pd.Series:
    times = pd.to_datetime(values, errors='raise')
    if times.isna().any() or getattr(times.dt, 'tz', None) is not None:
        raise ValueError('IBM AML timestamps must be complete and timezone-naive')
    return times


def time_boundaries(path: Path) -> tuple[pd.Timestamp, pd.Timestamp, int, int]:
    """Only timestamps participate in the boundary selection."""
    chunks = []
    for chunk in pd.read_csv(path, usecols=['Timestamp'], chunksize=CHUNK_SIZE):
        chunks.append(parse_time(chunk['Timestamp']).to_numpy(dtype='datetime64[ns]').view('int64'))
    if not chunks:
        raise ValueError('Empty IBM AML CSV')
    ordered = np.sort(np.concatenate(chunks))
    del chunks
    if len(ordered) < 100:
        raise ValueError('IBM AML CSV needs at least 100 rows')
    train_at = int(len(ordered) * .70)
    valid_at = int(len(ordered) * .85)
    train_cut = pd.Timestamp(ordered[train_at])
    valid_cut = pd.Timestamp(ordered[valid_at])
    if train_cut >= valid_cut:
        raise ValueError('Temporal boundaries overlap; insufficient time resolution')
    train_available = int(np.searchsorted(ordered, ordered[train_at], side='left'))
    return train_cut, valid_cut, len(ordered), train_available


def make_features(chunk: pd.DataFrame, times: pd.Series, history_features: pd.DataFrame | None = None) -> pd.DataFrame:
    if not chunk['Is Laundering'].isin([0, 1]).all():
        raise ValueError('Is Laundering must be binary')
    if not np.isfinite(chunk['Amount Paid'].to_numpy()).all() or (chunk['Amount Paid'] < 0).any():
        raise ValueError('Amount Paid must be finite and nonnegative')
    if chunk['From Bank'].isna().any() or chunk['To Bank'].isna().any():
        raise ValueError('Bank identifiers must be present to compute same_bank')
    amount = chunk['Amount Paid'].to_numpy(dtype='float32')
    features = pd.DataFrame({
        'amount_paid': amount,
        'log_amount_paid': np.log1p(amount).astype('float32'),
        'same_bank': chunk['From Bank'].eq(chunk['To Bank']).fillna(False).to_numpy(dtype='float32'),
        'same_currency': chunk['Payment Currency'].eq(chunk['Receiving Currency']).fillna(False).to_numpy(dtype='float32'),
        'hour': times.dt.hour.to_numpy(dtype='float32'),
        'weekday': times.dt.dayofweek.to_numpy(dtype='float32'),
        'receiving_currency': chunk['Receiving Currency'].astype('string').fillna('__missing__').to_numpy(),
        'payment_currency': chunk['Payment Currency'].astype('string').fillna('__missing__').to_numpy(),
        'payment_format': chunk['Payment Format'].astype('string').fillna('__missing__').to_numpy(),
    })
    if history_features is not None:
        if len(history_features) != len(features) or list(history_features.columns) != HISTORY_FEATURES:
            raise ValueError('invalid causal history feature frame')
        features = pd.concat([features.reset_index(drop=True), history_features.reset_index(drop=True)], axis=1)
    return features


def load_splits(path: Path, train_cut: pd.Timestamp, valid_cut: pd.Timestamp,
                train_available: int, max_train_rows: int | None, include_test: bool,
                feature_set: str = 'static_v1', history_max_accounts: int = 500_000):
    if feature_set not in {'static_v1', 'causal_history_v1'}:
        raise ValueError('feature_set must be static_v1 or causal_history_v1')
    parts: dict[str, list[pd.DataFrame]] = {key: [] for key in ('train', 'validation', 'test')}
    labels: dict[str, list[np.ndarray]] = {key: [] for key in parts}
    counts = {key: 0 for key in parts}
    train_seen = 0
    history_stream = (CausalHistoryStream(CausalAccountHistory(history_max_accounts))
                      if feature_set == 'causal_history_v1' else None)

    def consume(chunk: pd.DataFrame, history_features: pd.DataFrame | None = None):
        nonlocal train_seen
        times = parse_time(chunk['Timestamp'])
        masks = {'train': (times < train_cut).to_numpy(),
                 'validation': ((times >= train_cut) & (times < valid_cut)).to_numpy(),
                 'test': (times >= valid_cut).to_numpy()}
        for name, mask in masks.items():
            count = int(mask.sum())
            counts[name] += count
            if not count or (name == 'test' and not include_test):
                continue
            selected = np.flatnonzero(mask)
            if name == 'train' and max_train_rows is not None and train_available > max_train_rows:
                ranks = train_seen + np.arange(count, dtype=np.int64)
                keep = ((ranks + 1) * max_train_rows // train_available >
                        ranks * max_train_rows // train_available)
                selected = selected[keep]
            if len(selected):
                rows = chunk.iloc[selected]
                selected_history = history_features.iloc[selected] if history_features is not None else None
                parts[name].append(make_features(rows, times.iloc[selected], selected_history))
                labels[name].append(rows['Is Laundering'].to_numpy(dtype='int8'))
            if name == 'train':
                train_seen += count
    for chunk in iter_raw(path, include_accounts=history_stream is not None):
        if history_stream is None:
            consume(chunk)
        else:
            ready, historical = history_stream.feed(chunk)
            if not ready.empty:
                consume(ready, historical)
    if history_stream is not None:
        ready, historical = history_stream.finish()
        if not ready.empty:
            consume(ready, historical)
    result = {}
    for name in parts:
        if name == 'test' and not include_test:
            result[name] = None
            continue
        if not parts[name]:
            raise ValueError(f'Empty {name} split')
        x = pd.concat(parts[name], ignore_index=True)
        y = np.concatenate(labels[name])
        if len(np.unique(y)) != 2:
            raise ValueError(f'{name} split needs both labels')
        result[name] = (x, y)
    return result, counts


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int = 1000) -> float:
    k = min(k, len(y))
    order = np.argpartition(-scores, k - 1)[:k]
    return round(float(np.mean(y[order])), 6)


def train(data_path: Path, output: Path, mode: str = 'experiment', n_estimators: int = 180,
          max_train_rows: int | None = 500_000, max_false_alerts_per_10000: float | None = 100,
          feature_set: str = 'static_v1', history_max_accounts: int = 500_000):
    if mode not in {'experiment', 'final'} or n_estimators < 1:
        raise ValueError('Use experiment/final mode and positive estimators')
    if max_train_rows is not None and max_train_rows < 100:
        raise ValueError('max_train_rows must be at least 100')
    if feature_set not in {'static_v1', 'causal_history_v1'}:
        raise ValueError('feature_set must be static_v1 or causal_history_v1')
    if feature_set != 'static_v1' and mode == 'final':
        raise ValueError('causal history is a validation-only candidate; final mode is deliberately blocked')
    if output.exists() or output.with_suffix('.metrics.json').exists():
        raise FileExistsError('Output exists; preserve prior IBM AML experiments')
    validate_file(data_path)
    train_cut, valid_cut, total, train_available = time_boundaries(data_path)
    split, counts = load_splits(data_path, train_cut, valid_cut, train_available,
                                max_train_rows, mode == 'final', feature_set, history_max_accounts)
    x_train, y_train = split['train']
    x_valid, y_valid = split['validation']
    numeric_features = NUMERIC + (HISTORY_FEATURES if feature_set == 'causal_history_v1' else [])
    features = numeric_features + CATEGORICAL
    experiment, started = start_experiment(
        dataset='IBM AML HI-Small', algorithm='XGBoost', features=features,
        split_method='timestamp 70/15/15; final test omitted in experiment mode', mode=mode,
        hyperparameters={'n_estimators': n_estimators, 'max_train_rows': max_train_rows,
                         'max_false_alerts_per_10000': max_false_alerts_per_10000,
                         'max_depth': 4, 'learning_rate': .06, 'feature_set': feature_set,
                         'history_max_accounts': history_max_accounts if feature_set != 'static_v1' else None},
    )
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    pre = ColumnTransformer([('numeric', numeric, numeric_features), ('categorical', categorical, CATEGORICAL)])
    positive = int(y_train.sum())
    model = XGBClassifier(n_estimators=n_estimators, max_depth=4, learning_rate=.06,
                          subsample=.85, colsample_bytree=.85, reg_lambda=4,
                          objective='binary:logistic', tree_method='hist', eval_metric='aucpr',
                          scale_pos_weight=(len(y_train) - positive) / positive,
                          random_state=42, n_jobs=min(4, os.cpu_count() or 1))
    pipeline = Pipeline([('preprocessing', pre), ('model', model)])
    pipeline.fit(x_train, y_train)
    valid_scores = pipeline.predict_proba(x_valid)[:, 1]
    threshold, curve = threshold_analysis(y_valid, valid_scores, max_false_alerts_per_10000)
    if not 0 <= threshold <= 1:
        raise ValueError('Selected threshold cannot be serialized as a model threshold')
    validation_metrics = metrics(y_valid, valid_scores, threshold)
    validation_metrics['precision_at_1000'] = precision_at_k(y_valid, valid_scores)
    names = pre.get_feature_names_out()
    importance = sorted(zip(names, model.feature_importances_), key=lambda pair: float(pair[1]), reverse=True)
    report = {
        'format_version': 2, 'model_kind': 'ibm_aml', 'model_version': 'ibm-aml-hi-small-xgboost-v1',
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'IBM AML HI-Small synthetic laundering transactions',
        'data_files': [file_provenance(data_path)],
        'library_versions': {'python': sys.version.split()[0], 'pandas': pd.__version__,
                             'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__},
        'features': features, 'numerical_features': numeric_features, 'categorical_features': CATEGORICAL,
        'excluded_for_leakage': ['Is Laundering', 'Account', 'Account.1', 'Amount Received',
                                 'raw bank identifiers', 'future transaction history'],
        'split_method': 'timestamps below 70th percentile train, next 15% validation, last 15% final test; ties remain in later period',
        'split_boundaries': {'validation_start': train_cut.isoformat(), 'test_start': valid_cut.isoformat()},
        'split_counts': {**counts, 'total': total, 'train_used': len(y_train)},
        'experiment': finish_experiment(experiment, started),
        'threshold': float(threshold), 'threshold_analysis': curve,
        'validation_ranking_metrics': ranking_metrics(y_valid, valid_scores),
        'validation_operating_profiles': operating_profiles(y_valid, valid_scores, threshold),
        'validation_metrics': validation_metrics, 'test_metrics': None, 'baseline_test_metrics': None,
        'feature_importance': {'top_30': [{'feature': str(name), 'importance': round(float(value), 6)}
                                           for name, value in importance[:30]],
                               'leakage_audit': 'No target, account identifier, post-transfer received amount, or future history in X.'},
        'warning': 'Synthetic laundering labels; transaction-level score is not a calibrated bank fraud probability.',
    }
    if mode == 'final':
        x_test, y_test = split['test']
        test_scores = pipeline.predict_proba(x_test)[:, 1]
        report['test_metrics'] = metrics(y_test, test_scores, threshold)
        report['test_metrics']['precision_at_1000'] = precision_at_k(y_test, test_scores)
        baseline_size = min(len(y_train), 300_000)
        baseline_index = np.linspace(0, len(y_train) - 1, baseline_size, dtype=np.int64)
        baseline_pre = ColumnTransformer([
            ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')),
                                  ('scale', StandardScaler(with_mean=False))]), numeric_features),
            ('categorical', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
                                      ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL),
        ])
        baseline = Pipeline([('preprocessing', baseline_pre),
                             ('model', LogisticRegression(solver='liblinear', max_iter=500, tol=1e-3,
                                                          class_weight='balanced'))])
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=ConvergenceWarning)
            baseline.fit(x_train.iloc[baseline_index], y_train[baseline_index])
        report['baseline_train_used'] = baseline_size
        report['baseline_convergence'] = {'iterations': int(baseline.named_steps['model'].n_iter_[0]),
                                          'max_iter': 500, 'solver': 'liblinear'}
        report['baseline_test_metrics'] = metrics(y_test, baseline.predict_proba(x_test)[:, 1], .5)
        artifact = dict(report)
        artifact['pipeline'] = pipeline
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'mode': mode, 'validation': validation_metrics,
                      'test': report['test_metrics'], 'output': str(output)}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['experiment', 'final'], default='experiment')
    parser.add_argument('--estimators', type=int, default=180)
    parser.add_argument('--max-train-rows', type=int, default=500_000)
    parser.add_argument('--full-train', action='store_true', help='Explicitly fit all early-period rows')
    parser.add_argument('--max-false-alerts-per-10000', type=float, default=100)
    parser.add_argument('--feature-set', choices=['static_v1', 'causal_history_v1'], default='static_v1')
    parser.add_argument('--history-max-accounts', type=int, default=500_000)
    args = parser.parse_args()
    train(args.data, args.output, args.mode, args.estimators,
          None if args.full_train else args.max_train_rows, args.max_false_alerts_per_10000,
          args.feature_set, args.history_max_accounts)


if __name__ == '__main__':
    main()
