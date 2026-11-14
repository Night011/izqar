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
          model_version: str = 'paysim-xgboost-v1', mode: str = 'final', positive_weight_multiplier: float = 1.0):
    if output.exists() or output.with_suffix('.metrics.json').exists():
        raise FileExistsError(f'Output already exists; choose a new run path to preserve prior results: {output}')
    if not model_version.startswith('paysim-xgboost-v') or not model_version.removeprefix('paysim-xgboost-v').isdigit():
        raise ValueError('model_version must have the form paysim-xgboost-vN')
    if mode not in {'experiment', 'final'}:
        raise ValueError('mode must be experiment or final')
    if not np.isfinite(positive_weight_multiplier) or positive_weight_multiplier <= 0:
        raise ValueError('positive_weight_multiplier must be a finite positive number')
    if mode == 'final' and positive_weight_multiplier != 1.0:
        raise ValueError('positive weight candidates are validation-only; final mode requires multiplier 1.0')
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
        dataset='PaySim', algorithm='XGBoost', features=FEATURES,
        split_method='chronological step 70/15/15; final test omitted in experiment mode', mode=mode,
        hyperparameters={'n_estimators': n_estimators, 'max_rows': max_rows, 'max_depth': 4,
                         'learning_rate': .065, 'feature_set': 'pre_transaction_v1',
                         'positive_weight_multiplier': positive_weight_multiplier},
    )
    transform = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', add_indicator=True), NUMERIC),
        ('category', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
    ])
    y_train = train_rows['isFraud'].astype(int)
    positives, negatives = int(y_train.sum()), int(len(y_train) - y_train.sum())
    baseline_transform = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scale', StandardScaler(with_mean=False)),
                              ('clip', FunctionTransformer(np.clip, kw_args={'a_min': -20, 'a_max': 20}))]), NUMERIC),
        ('category', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
    ])
    baseline = Pipeline([('preprocessing', baseline_transform),
                         ('model', LogisticRegression(max_iter=1000, tol=1e-3, class_weight='balanced', solver='liblinear'))])
    model = XGBClassifier(n_estimators=n_estimators, max_depth=4, learning_rate=.065, subsample=.85, colsample_bytree=.85,
                          reg_lambda=3, objective='binary:logistic', tree_method='hist', eval_metric='aucpr', random_state=42,
                          n_jobs=min(4, os.cpu_count() or 1), scale_pos_weight=(negatives / positives) * positive_weight_multiplier)
    pipeline = Pipeline([('preprocessing', transform), ('model', model)])
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=ConvergenceWarning)
        baseline.fit(train_rows[FEATURES], y_train)
    if baseline.named_steps['model'].n_iter_[0] >= 1000:
        raise RuntimeError('PaySim baseline logistic regression did not converge')
    pipeline.fit(train_rows[FEATURES], y_train)
    encoded_features = pipeline.named_steps['preprocessing'].get_feature_names_out()
    importances = pipeline.named_steps['model'].feature_importances_
    ranked = sorted(zip(encoded_features, importances), key=lambda item: item[1], reverse=True)
    valid_scores = pipeline.predict_proba(valid_rows[FEATURES])[:, 1]
    threshold = choose_threshold(valid_rows['isFraud'].astype(int), valid_scores)
    test_metrics = None
    baseline_test_metrics = None
    if mode == 'final':
        test_scores = pipeline.predict_proba(test_rows[FEATURES])[:, 1]
        test_metrics = evaluate(test_rows['isFraud'].astype(int), test_scores, threshold)
        baseline_test_metrics = evaluate(test_rows['isFraud'].astype(int), baseline.predict_proba(test_rows[FEATURES])[:, 1], .5)
    artifact = {
        'format_version': 2, 'model_kind': 'paysim', 'model_version': model_version,
        'trained_at_utc': datetime.now(timezone.utc).isoformat(), 'dataset': 'PaySim (synthetic Kaggle dataset)',
        'data_files': [file_provenance(data_path)],
        'library_versions': {'python': sys.version.split()[0], 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__},
        'split_counts': {'train': len(train_rows), 'validation': len(valid_rows), 'test': len(test_rows)},
        'experiment': finish_experiment(experiment, started),
        'features': FEATURES, 'numerical_features': NUMERIC, 'categorical_features': CATEGORICAL, 'pipeline': pipeline,
        'threshold': threshold, 'validation_metrics': evaluate(valid_rows['isFraud'].astype(int), valid_scores, threshold),
        'validation_calibration': calibration_diagnostics(valid_rows['isFraud'].astype(int), valid_scores),
        'validation_ranking_metrics': ranking_metrics(valid_rows['isFraud'].astype(int), valid_scores),
        'validation_operating_profiles': operating_profiles(valid_rows['isFraud'].astype(int), valid_scores, threshold),
        'test_metrics': test_metrics,
        'baseline_test_metrics': baseline_test_metrics,
        'baseline_convergence': {'solver': 'liblinear', 'iterations': int(baseline.named_steps['model'].n_iter_[0]), 'max_iter': 1000, 'converged': True},
        'feature_importance': {'top_30': [{'feature': str(name), 'importance': round(float(value), 6)} for name, value in ranked[:30]],
                               'leakage_audit': 'Encoded features derive only from type, amount, oldbalanceOrg and oldbalanceDest; importance does not establish causality.'},
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
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows < 100:
        parser.error('--max-rows must be at least 100')
    train(args.data, args.output, args.estimators, None if args.full_data else args.max_rows,
          args.model_version, args.mode, args.positive_weight_multiplier)


if __name__ == '__main__':
    main()
