"""Train a leak-resistant PaySim classifier using only pre-transaction fields.

The original CSV is never modified. `isFraud` is only the training target;
`isFlaggedFraud`, post-transaction balances, participant IDs and `step` are
not model features. Splits are chronological by step (70/15/15).
"""
from __future__ import annotations

import argparse
import hashlib
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
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from experiment_tracking import finish_experiment, start_experiment
from train_ieee import calibration_diagnostics, operating_profiles, ranking_metrics

FEATURES = ['type', 'amount', 'oldbalanceOrg', 'oldbalanceDest']
NUMERIC = ['amount', 'oldbalanceOrg', 'oldbalanceDest']
CATEGORICAL = ['type']
REQUIRED = {'step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFraud'}
ENGINEERED_NUMERIC = ['amount_to_origin_balance', 'amount_to_destination_balance',
                      'destination_balance_is_zero']


def feature_schema(feature_set: str) -> tuple[list[str], list[str], list[str]]:
    """Return a decision-time feature contract for a named PaySim experiment."""
    if feature_set == 'pre_transaction_v1':
        return FEATURES, NUMERIC, CATEGORICAL
    if feature_set == 'pre_transaction_v2':
        return FEATURES + ENGINEERED_NUMERIC, NUMERIC + ENGINEERED_NUMERIC, CATEGORICAL
    raise ValueError('feature_set must be pre_transaction_v1 or pre_transaction_v2')


def decision_time_features(rows: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Derive candidate features only from the four API-available input fields.

    No label, post-event balance (including an arithmetic proxy for it),
    participant ID, time step, or cross-row history is read here. Ratios use
    one as a lower denominator bound so a zero opening balance is represented
    without infinity.
    """
    features, _, _ = feature_schema(feature_set)
    missing = set(FEATURES) - set(rows.columns)
    if missing:
        raise ValueError(f'Missing decision-time fields: {sorted(missing)}')
    result = rows[FEATURES].copy()
    if feature_set == 'pre_transaction_v2':
        amount = pd.to_numeric(result['amount'], errors='coerce')
        origin = pd.to_numeric(result['oldbalanceOrg'], errors='coerce')
        destination = pd.to_numeric(result['oldbalanceDest'], errors='coerce')
        result['amount_to_origin_balance'] = amount / origin.clip(lower=1.0)
        result['amount_to_destination_balance'] = amount / destination.clip(lower=1.0)
        result['destination_balance_is_zero'] = np.where(destination.isna(), np.nan, (destination == 0).astype('float32'))
    return result.loc[:, features]


def validate_run_configuration(mode: str, positive_weight_multiplier: float, feature_set: str,
                               frozen_final_feature_set: str | None) -> None:
    """Protect the held-out test from accidental candidate iteration."""
    if mode == 'final' and positive_weight_multiplier != 1.0:
        raise ValueError('positive weight candidates are validation-only; final mode requires multiplier 1.0')
    if frozen_final_feature_set is not None and mode != 'final':
        raise ValueError('--frozen-final-feature-set is valid only in final mode')
    if frozen_final_feature_set is not None and frozen_final_feature_set != feature_set:
        raise ValueError('frozen final feature set must exactly match --feature-set')
    if mode == 'final' and feature_set != 'pre_transaction_v1' and frozen_final_feature_set != feature_set:
        raise ValueError('engineered feature candidates require an explicit matching frozen final feature set')


def file_provenance(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'filename': path.name, 'size_bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def evaluate(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'average_precision_pr_auc': round(float(average_precision_score(y_true, scores)), 6),
        'roc_auc': round(float(roc_auc_score(y_true, scores)), 6),
        'precision': round(float(precision_score(y_true, pred, zero_division=0)), 6),
        'recall': round(float(recall_score(y_true, pred, zero_division=0)), 6),
        'f1': round(float(f1_score(y_true, pred, zero_division=0)), 6),
        'confusion_matrix_tn_fp_fn_tp': [int(tn), int(fp), int(fn), int(tp)],
        'false_positive_rate': round(float(fp / max(tn + fp, 1)), 6),
        'false_negative_rate': round(float(fn / max(fn + tp, 1)), 6),
        'false_alerts_per_10000_operations': round(float(fp * 10000 / max(len(y_true), 1)), 3),
        'cases': int(len(y_true)), 'fraud_cases': int(y_true.sum()),
    }


def choose_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def type_slice_metrics(rows: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict:
    """Validation-only quality slices for transfer types that can contain fraud.

    ``type`` is known before the transaction and is already a serving feature.
    This function never uses an account identifier or a post-transaction field;
    the label is used solely to measure the saved validation prediction.
    """
    result = {}
    for transaction_type in ('TRANSFER', 'CASH_OUT'):
        mask = rows['type'].astype(str).eq(transaction_type).to_numpy()
        y = rows.loc[mask, 'isFraud'].astype(int).to_numpy()
        if len(y) == 0:
            result[transaction_type] = {'available': False, 'reason': 'type absent from validation window'}
        elif len(np.unique(y)) < 2:
            result[transaction_type] = {
                'available': False, 'reason': 'slice has only one label class; ranking metrics are undefined',
                'cases': int(len(y)), 'fraud_cases': int(y.sum()),
            }
        else:
            result[transaction_type] = {'available': True, **evaluate(y, scores[mask], threshold)}
    return result


def robustness_diagnostics(pipeline: Pipeline, rows: pd.DataFrame, scores: np.ndarray, threshold: float,
                           feature_set: str) -> dict:
    """Exercise serving preprocessing under explicit, non-production perturbations.

    These diagnostics are intentionally validation-only and are not a claim
    about future bank traffic.  Missingness is deterministic so reports are
    reproducible.  Unknown-category handling measures schema resilience, not
    fraud quality, because the synthetic category has no truthful label.
    """
    features = rows[FEATURES].copy()
    y = rows['isFraud'].astype(int).to_numpy()
    missing = features.copy()
    missing_positions = np.arange(len(missing)) % 10 == 0
    missing.loc[missing.index[missing_positions], ['oldbalanceOrg', 'oldbalanceDest']] = np.nan
    missing_scores = pipeline.predict_proba(decision_time_features(missing, feature_set))[:, 1]
    unknown = features.copy()
    unknown['type'] = '__unseen_transaction_type__'
    unknown_scores = pipeline.predict_proba(decision_time_features(unknown, feature_set))[:, 1]
    return {
        'scope': 'validation_only_synthetic_preprocessing_stress',
        'warning': 'Perturbations are diagnostic only; they do not model a future fraud distribution or authorize threshold changes.',
        'missing_balances_every_tenth_row': {
            'affected_rows': int(missing_positions.sum()), 'metrics': evaluate(y, missing_scores, threshold),
            'recall_delta_vs_unmodified': round(float(recall_score(y, (missing_scores >= threshold).astype(int), zero_division=0)
                                                    - recall_score(y, (scores >= threshold).astype(int), zero_division=0)), 6),
        },
        'unseen_type_encoding': {
            'synthetic_category': '__unseen_transaction_type__', 'cases': int(len(unknown)),
            'mean_score_delta_vs_unmodified': round(float(np.mean(unknown_scores) - np.mean(scores)), 6),
            'flagged_rate_unmodified': round(float(np.mean(scores >= threshold)), 6),
            'flagged_rate_with_unseen_type': round(float(np.mean(unknown_scores >= threshold)), 6),
        },
    }


def read_paysim(path: Path, max_rows: int | None) -> pd.DataFrame:
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = REQUIRED - columns
    if missing:
        raise ValueError(f'Missing required PaySim columns: {sorted(missing)}')
    usecols = sorted(REQUIRED)
    # nrows prevents accidental full-memory loads during developer experiments.
    data = pd.read_csv(path, usecols=usecols, nrows=max_rows, low_memory=False,
                       dtype={'step': 'int32', 'amount': 'float32', 'oldbalanceOrg': 'float32', 'oldbalanceDest': 'float32', 'isFraud': 'int8', 'type': 'category'})
    if not data['isFraud'].isin([0, 1]).all():
        raise ValueError('isFraud must be binary')
    return data.sort_values('step', kind='stable').reset_index(drop=True)


def train(data_path: Path, output: Path, n_estimators: int, max_rows: int | None,
          model_version: str = 'paysim-xgboost-v1', mode: str = 'final', positive_weight_multiplier: float = 1.0,
          feature_set: str = 'pre_transaction_v1', frozen_final_feature_set: str | None = None):
    if output.exists() or output.with_suffix('.metrics.json').exists():
        raise FileExistsError(f'Output already exists; choose a new run path to preserve prior results: {output}')
    if not model_version.startswith('paysim-xgboost-v') or not model_version.removeprefix('paysim-xgboost-v').isdigit():
        raise ValueError('model_version must have the form paysim-xgboost-vN')
    if mode not in {'experiment', 'final'}:
        raise ValueError('mode must be experiment or final')
    if not np.isfinite(positive_weight_multiplier) or positive_weight_multiplier <= 0:
        raise ValueError('positive_weight_multiplier must be a finite positive number')
    features, numeric, categorical = feature_schema(feature_set)
    validate_run_configuration(mode, positive_weight_multiplier, feature_set, frozen_final_feature_set)
    if not data_path.is_file():
        raise FileNotFoundError(f'Missing {data_path}. Download PaySim manually from Kaggle and keep it outside Git.')
    data = read_paysim(data_path, max_rows)
    if len(data) < 100:
        raise ValueError('Need at least 100 rows for chronological training and evaluation')
    first, second = data['step'].iloc[int(len(data) * .70)], data['step'].iloc[int(len(data) * .85)]
    train_rows, valid_rows, test_rows = data[data.step < first], data[(data.step >= first) & (data.step < second)], data[data.step >= second]
    for label, part in [('train', train_rows), ('validation', valid_rows), ('test', test_rows)]:
        if part.empty or part['isFraud'].nunique() != 2:
            raise ValueError(f'{label} split must contain both classes. Increase --max-rows or use the full chronology.')
    experiment, started = start_experiment(
        dataset='PaySim', algorithm='XGBoost', features=features,
        split_method='chronological step 70/15/15; final test omitted in experiment mode', mode=mode,
        hyperparameters={'n_estimators': n_estimators, 'max_rows': max_rows, 'max_depth': 4,
                         'learning_rate': .065, 'feature_set': feature_set,
                         'positive_weight_multiplier': positive_weight_multiplier,
                         'frozen_final_feature_set': frozen_final_feature_set},
    )
    transform = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', add_indicator=True), numeric),
        ('category', OneHotEncoder(handle_unknown='ignore'), categorical),
    ])
    y_train = train_rows['isFraud'].astype(int)
    positives, negatives = int(y_train.sum()), int(len(y_train) - y_train.sum())
    baseline_transform = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scale', StandardScaler(with_mean=False)),
                              ('clip', FunctionTransformer(np.clip, kw_args={'a_min': -20, 'a_max': 20}))]), numeric),
        ('category', OneHotEncoder(handle_unknown='ignore'), categorical),
    ])
    baseline = Pipeline([('preprocessing', baseline_transform),
                         ('model', LogisticRegression(max_iter=1000, tol=1e-3, class_weight='balanced', solver='liblinear'))])
    model = XGBClassifier(n_estimators=n_estimators, max_depth=4, learning_rate=.065, subsample=.85, colsample_bytree=.85,
                          reg_lambda=3, objective='binary:logistic', tree_method='hist', eval_metric='aucpr', random_state=42,
                          n_jobs=min(4, os.cpu_count() or 1), scale_pos_weight=(negatives / positives) * positive_weight_multiplier)
    pipeline = Pipeline([('preprocessing', transform), ('model', model)])
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=ConvergenceWarning)
        baseline.fit(decision_time_features(train_rows, feature_set), y_train)
    if baseline.named_steps['model'].n_iter_[0] >= 1000:
        raise RuntimeError('PaySim baseline logistic regression did not converge')
    pipeline.fit(decision_time_features(train_rows, feature_set), y_train)
    encoded_features = pipeline.named_steps['preprocessing'].get_feature_names_out()
    importances = pipeline.named_steps['model'].feature_importances_
    ranked = sorted(zip(encoded_features, importances), key=lambda item: item[1], reverse=True)
    valid_scores = pipeline.predict_proba(decision_time_features(valid_rows, feature_set))[:, 1]
    threshold = choose_threshold(valid_rows['isFraud'].astype(int), valid_scores)
    test_metrics = None
    baseline_test_metrics = None
    if mode == 'final':
        test_scores = pipeline.predict_proba(decision_time_features(test_rows, feature_set))[:, 1]
        test_metrics = evaluate(test_rows['isFraud'].astype(int), test_scores, threshold)
        baseline_test_metrics = evaluate(test_rows['isFraud'].astype(int), baseline.predict_proba(decision_time_features(test_rows, feature_set))[:, 1], .5)
    artifact = {
        'format_version': 2, 'model_kind': 'paysim', 'model_version': model_version,
        'run_mode': mode,
        'trained_at_utc': datetime.now(timezone.utc).isoformat(), 'dataset': 'PaySim (synthetic Kaggle dataset)',
        'data_files': [file_provenance(data_path)],
        'library_versions': {'python': sys.version.split()[0], 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__},
        'split_counts': {'train': len(train_rows), 'validation': len(valid_rows), 'test': len(test_rows)},
        'experiment': finish_experiment(experiment, started),
        'features': features, 'numerical_features': numeric, 'categorical_features': categorical, 'pipeline': pipeline,
        'threshold': threshold, 'validation_metrics': evaluate(valid_rows['isFraud'].astype(int), valid_scores, threshold),
        'validation_type_slices': type_slice_metrics(valid_rows, valid_scores, threshold),
        'validation_robustness': robustness_diagnostics(pipeline, valid_rows, valid_scores, threshold, feature_set),
        'validation_calibration': calibration_diagnostics(valid_rows['isFraud'].astype(int), valid_scores),
        'validation_ranking_metrics': ranking_metrics(valid_rows['isFraud'].astype(int), valid_scores),
        'validation_operating_profiles': operating_profiles(valid_rows['isFraud'].astype(int), valid_scores, threshold),
        'test_metrics': test_metrics,
        'baseline_test_metrics': baseline_test_metrics,
        'baseline_convergence': {'solver': 'liblinear', 'iterations': int(baseline.named_steps['model'].n_iter_[0]), 'max_iter': 1000, 'converged': True},
        'feature_importance': {'top_30': [{'feature': str(name), 'importance': round(float(value), 6)} for name, value in ranked[:30]],
                               'leakage_audit': ('Encoded features derive only from type, amount, oldbalanceOrg and oldbalanceDest; '
                                                 'engineered ratios and flags exclude arithmetic post-balance proxies; '
                                                 'importance does not establish causality.')},
        'split_method': '70/15/15 chronological step ascending',
        'excluded_for_leakage': ['isFraud', 'isFlaggedFraud', 'newbalanceOrig', 'newbalanceDest', 'nameOrig', 'nameDest', 'step'],
        'warning': 'PaySim is synthetic; output is an uncalibrated research-model score, not a banking fraud probability.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'final':
        joblib.dump(artifact, output, compress=3)
    report = {key: value for key, value in artifact.items() if key != 'pipeline'}
    report['threshold'] = round(float(threshold), 6)
    report_path = output.with_suffix('.metrics.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'model_path': str(output) if mode == 'final' else None, 'metrics_path': str(report_path),
                      'validation': artifact['validation_metrics'], 'test': artifact['test_metrics']}, ensure_ascii=False, indent=2))
    if mode == 'final':
        print('IMPORTANT: isFraud was used only as y; the held-out test set was evaluated once after validation threshold selection.')
    return report


def main():
    parser = argparse.ArgumentParser(description='Train PaySim XGBoost with leak-resistant chronological splits')
    parser.add_argument('--data', type=Path, required=True, help='Path to original PS_20174392719_1491204439457_log.csv')
    parser.add_argument('--output', type=Path, default=Path('models/paysim_model.joblib'))
    parser.add_argument('--estimators', type=int, default=180)
    parser.add_argument('--max-rows', type=int, default=500_000, help='Default memory cap: 500000 chronological CSV rows')
    parser.add_argument('--full-data', action='store_true', help='Explicitly opt into loading the entire CSV in RAM')
    parser.add_argument('--mode', choices=['experiment', 'final'], default='experiment',
                        help='Experiment mode omits the held-out test and writes metrics only')
    parser.add_argument('--model-version', default='paysim-xgboost-v1', help='Version identifier saved with the artifact')
    parser.add_argument('--positive-weight-multiplier', type=float, default=1.0,
                        help='Experiment-only multiplier over inverse-prevalence class weight')
    parser.add_argument('--feature-set', choices=['pre_transaction_v1', 'pre_transaction_v2'], default='pre_transaction_v1',
                        help='v2 adds only row-local transformations of API-available pre-transaction fields')
    parser.add_argument('--frozen-final-feature-set', choices=['pre_transaction_v2'],
                        help='Required exact acknowledgement before a sealed v2 final run')
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 100:
        parser.error('--max-rows must be at least 100')
    train(args.data, args.output, args.estimators, None if args.full_data else args.max_rows,
          args.model_version, args.mode, args.positive_weight_multiplier, args.feature_set,
          args.frozen_final_feature_set)


if __name__ == '__main__':
    main()
