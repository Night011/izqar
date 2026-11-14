"""Train a reproducible IEEE-CIS transaction classifier with a chronological holdout.

This model NEVER trains on the synthetic transfer simulator.
It doesn't imply real-world fraud performance or calibrated probabilities.
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

NUMERIC = [
    'TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2',
    'dist1', 'dist2', 'C1', 'C2', 'C4', 'C6', 'C7', 'C8', 'C9', 'C10',
    'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D15',
    'V1', 'V2', 'V3',
]
EXTENDED_NUMERIC = ['C3', 'C5', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
CATEGORICAL = [
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'DeviceType', 'DeviceInfo', 'id_31',
]
FORBIDDEN = {'isFraud', 'TransactionID', 'TransactionDT'}


def split_chronologically(data: pd.DataFrame):
    """Boundary timestamps cannot straddle train, validation, or final test."""
    data = data.sort_values(['TransactionDT', 'TransactionID'], kind='stable').reset_index(drop=True)
    if len(data) < 100:
        raise ValueError('Need at least 100 labeled rows')
    validation_start = data['TransactionDT'].iloc[int(len(data) * .70)]
    test_start = data['TransactionDT'].iloc[int(len(data) * .85)]
    parts = (data[data.TransactionDT < validation_start],
             data[(data.TransactionDT >= validation_start) & (data.TransactionDT < test_start)],
             data[data.TransactionDT >= test_start])
    for label, part in zip(('train', 'validation'), parts[:2]):
        if part.empty or part['isFraud'].nunique() != 2:
            raise ValueError(f'{label} split must contain both classes; use the full labeled CSV')
    if parts[2].empty:
        raise ValueError('Final test window is empty')
    return parts, {'validation_start_dt': float(validation_start), 'test_start_dt': float(test_start)}


def threshold_analysis(y_true, scores, max_false_alerts_per_10000: float | None = None):
    """Select using validation labels only; report a compact operating curve."""
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) != len(s) or not len(y) or not np.isfinite(s).all():
        raise ValueError('Validation labels and finite scores of equal nonzero length required')
    precision, recall, thresholds = precision_recall_curve(y, s)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_f1 = float(thresholds[int(np.argmax(f1))])
    selected = best_f1
    if max_false_alerts_per_10000 is not None:
        if max_false_alerts_per_10000 < 0:
            raise ValueError('False-alert budget must be nonnegative')
        # Exact unique thresholds, including an all-negative operating point.
        order = np.argsort(-s, kind='stable')
        sorted_scores, sorted_y = s[order], y[order]
        tp = np.cumsum(sorted_y)
        fp = np.cumsum(1 - sorted_y)
        ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(s) - 1]
        feasible = ends[fp[ends] * 10000 / len(y) <= max_false_alerts_per_10000]
        selected = float(sorted_scores[feasible[np.argmax(tp[feasible])]]) if len(feasible) else float(np.nextafter(sorted_scores[0], np.inf))
    sampled = np.unique(np.linspace(0, len(thresholds) - 1, min(101, len(thresholds)), dtype=int))
    curve = []
    for index in sampled:
        threshold = float(thresholds[index])
        pred = s >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum((~pred) & (y == 1)))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        curve.append({'threshold': round(threshold, 6), 'precision': round(p, 6), 'recall': round(r, 6),
                      'f1': round(2 * p * r / max(p + r, 1e-12), 6), 'false_alerts': fp,
                      'false_alerts_per_10000': round(fp * 10000 / len(y), 3)})
    return selected, {'max_f1_threshold': best_f1, 'selected_threshold': selected,
                      'strategy': 'false_alert_budget' if max_false_alerts_per_10000 is not None else 'max_f1',
                      'max_false_alerts_per_10000': max_false_alerts_per_10000, 'curve': curve}


def ranking_metrics(y_true, scores, ks: tuple[int, ...] = (100, 500, 1000)) -> dict:
    """Ranking quality for a bounded investigation queue; no threshold tuning."""
    y, values = np.asarray(y_true, dtype=np.int8), np.asarray(scores, dtype=float)
    if len(y) != len(values) or not len(y) or not np.isfinite(values).all():
        raise ValueError('ranking metrics need equally sized finite labels and scores')
    total_positive = int(y.sum())
    order = np.argsort(-values, kind='stable')
    results = {}
    for requested in ks:
        if requested < 1:
            raise ValueError('ranking cutoffs must be positive')
        actual = min(int(requested), len(y))
        found = int(y[order[:actual]].sum())
        results[str(requested)] = {
            'alerts': actual,
            'precision_at_k': round(found / actual, 6),
            'recall_at_k': round(found / max(total_positive, 1), 6),
            'true_positives_at_k': found,
        }
    return results


def operating_profiles(y_true, scores, analyst_threshold: float,
                       monitoring_min_recall: float = .80,
                       high_confidence_min_precision: float = .80) -> dict:
    """Choose three named operating points from validation only.

    A PR curve is sampled instead of evaluating every unique threshold, keeping
    memory and CPU bounded on the large IEEE/IBM validation windows.  Profiles
    that cannot meet their constraint are reported as unavailable rather than
    silently returning an attractive but invalid operating point.
    """
    if not 0 <= analyst_threshold <= 1 or not 0 < monitoring_min_recall <= 1 or not 0 < high_confidence_min_precision <= 1:
        raise ValueError('invalid operating-profile constraint')
    y, values = np.asarray(y_true, dtype=np.int8), np.asarray(scores, dtype=float)
    if len(y) != len(values) or not len(y) or not np.isfinite(values).all():
        raise ValueError('operating profiles need equally sized finite labels and scores')
    precision, recall, thresholds = precision_recall_curve(y, values)
    candidate_precision, candidate_recall = precision[:-1], recall[:-1]
    if not len(thresholds):
        raise ValueError('operating profiles need score variation')

    def materialize(index: int, constraint: str, met: bool) -> dict:
        threshold = float(thresholds[index])
        return {'threshold': round(threshold, 6), 'constraint': constraint,
                'constraint_met': met, 'metrics': metrics(y, values, threshold)}

    profiles = {'analyst_review': materialize(int(np.argmin(np.abs(thresholds - analyst_threshold))),
                                               'validation-selected review threshold', True)}
    monitoring = np.flatnonzero(candidate_recall >= monitoring_min_recall)
    if len(monitoring):
        # Largest feasible threshold minimises review load while retaining the target recall.
        profiles['monitoring'] = materialize(int(monitoring[-1]), f'recall >= {monitoring_min_recall:.2f}', True)
    else:
        best = int(np.argmax(candidate_recall))
        profiles['monitoring'] = materialize(best, f'recall >= {monitoring_min_recall:.2f}', False)

    confident = np.flatnonzero(candidate_precision >= high_confidence_min_precision)
    if len(confident):
        best_recall = candidate_recall[confident].max()
        best = int(confident[np.flatnonzero(candidate_recall[confident] == best_recall)[-1]])
        profiles['high_confidence'] = materialize(best, f'precision >= {high_confidence_min_precision:.2f}', True)
    else:
        profiles['high_confidence'] = {'available': False,
                                       'constraint': f'precision >= {high_confidence_min_precision:.2f}',
                                       'reason': 'No validation threshold meets this precision constraint.'}
    return profiles


def feature_importance(pipeline: Pipeline, features: list[str]) -> dict:
    names = pipeline.named_steps['preprocessing'].get_feature_names_out()
    values = pipeline.named_steps['model'].feature_importances_
    ranked = sorted(zip(names, values), key=lambda pair: float(pair[1]), reverse=True)
    suspicious = [name for name in names if any(term.lower() in name.lower() for term in FORBIDDEN)]
    if suspicious or FORBIDDEN.intersection(features):
        raise ValueError(f'Potential target/identifier leakage: {suspicious}')
    return {'top_30': [{'feature': name, 'importance': round(float(value), 6)} for name, value in ranked[:30]],
            'method': 'XGBoost built-in feature importance (gain-based); association does not establish causal validity',
            'leakage_audit': 'No target, TransactionID, or TransactionDT in model features; temporal availability of anonymized features still requires domain review.'}


def file_provenance(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'filename': path.name, 'size_bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def metrics(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'average_precision_pr_auc': round(float(average_precision_score(y_true, scores)), 6),
        'roc_auc': round(float(roc_auc_score(y_true, scores)), 6),
        'precision': round(float(precision_score(y_true, pred, zero_division=0)), 6),
        'recall': round(float(recall_score(y_true, pred, zero_division=0)), 6),
        'f1': round(float(f1_score(y_true, pred, zero_division=0)), 6),
        'confusion_matrix_tn_fp_fn_tp': [int(tn), int(fp), int(fn), int(tp)],
        'false_positive_rate': round(float(fp / max(fp + tn, 1)), 6),
        'false_negative_rate': round(float(fn / max(fn + tp, 1)), 6),
        'false_alerts_per_10000_operations': round(float(fp * 10000 / max(len(y_true), 1)), 3),
        'cases': int(len(y_true)),
        'fraud_cases': int(y_true.sum()),
    }


def choose_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # The last precision/recall point has no corresponding threshold.
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def train(data_dir: Path, output: Path, n_estimators: int, max_rows: int | None = None,
          mode: str = 'final', max_false_alerts_per_10000: float | None = None,
          feature_set: str = 'core', positive_weight_multiplier: float = 1.0):
    if mode not in {'experiment', 'final'} or n_estimators < 1:
        raise ValueError('mode must be experiment or final; estimators must be positive')
    if feature_set not in {'core', 'extended_cv'}:
        raise ValueError('feature_set must be core or extended_cv')
    if mode == 'final' and feature_set != 'core':
        raise ValueError('The existing final test has been inspected; extended_cv is validation-only until a new independent holdout is available')
    if output.exists() or output.with_suffix('.metrics.json').exists():
        raise FileExistsError(f'Output already exists; choose a new run path to preserve prior results: {output}')
    tx_path = data_dir / 'train_transaction.csv'
    id_path = data_dir / 'train_identity.csv'
    if not tx_path.exists():
        raise FileNotFoundError(f'Missing {tx_path}. Download IEEE-CIS from Kaggle.')
    tx_columns = set(pd.read_csv(tx_path, nrows=0).columns)
    required = {'TransactionID', 'TransactionDT', 'TransactionAmt', 'isFraud'}
    if missing := (required - tx_columns):
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    requested_numeric = NUMERIC + (EXTENDED_NUMERIC if feature_set == 'extended_cv' else [])
    tx_usecols = sorted((tx_columns & set(requested_numeric + CATEGORICAL)) | required)
    numeric_columns = set(requested_numeric) & set(tx_usecols)
    dtype = {name: 'float32' for name in numeric_columns}
    dtype.update({'TransactionID': 'int64', 'TransactionDT': 'float64', 'isFraud': 'int8'})
    data = pd.read_csv(tx_path, usecols=tx_usecols, dtype=dtype, low_memory=False)
    if data['TransactionID'].isna().any() or not data['TransactionID'].is_unique:
        raise ValueError('TransactionID must be non-null and unique in train_transaction.csv')
    if data['TransactionDT'].isna().any() or not data['isFraud'].isin([0, 1]).all():
        raise ValueError('TransactionDT must be present and isFraud must be binary')

    if not id_path.is_file():
        raise FileNotFoundError(f'Missing {id_path}; IEEE-CIS training requires both transaction and identity CSV files.')
    identity_cols = set(pd.read_csv(id_path, nrows=0).columns)
    if 'TransactionID' not in identity_cols:
        raise ValueError('train_identity.csv lacks TransactionID')
    usecols = sorted(identity_cols & (set(CATEGORICAL) | {'TransactionID'}))
    identity_parts = []
    seen_ids = set()
    relevant_ids = set(data['TransactionID'])
    for chunk in pd.read_csv(id_path, usecols=usecols, chunksize=100_000, low_memory=False):
        if chunk['TransactionID'].isna().any() or chunk['TransactionID'].duplicated().any() or seen_ids.intersection(chunk['TransactionID']):
            raise ValueError('TransactionID must be non-null and unique in train_identity.csv')
        seen_ids.update(chunk['TransactionID'])
        selected = chunk[chunk['TransactionID'].isin(relevant_ids)]
        if not selected.empty:
            identity_parts.append(selected)
    if identity_parts:
        data = data.merge(pd.concat(identity_parts, ignore_index=True), on='TransactionID', how='left', validate='one_to_one')
    elif len(usecols) > 1:
        for column in usecols:
            if column != 'TransactionID':
                data[column] = pd.NA
    nums = [column for column in requested_numeric if column in data]
    cats = [column for column in CATEGORICAL if column in data]
    features = nums + cats
    for col in nums:
        data[col] = pd.to_numeric(data[col], errors='coerce').astype('float32')
    for col in cats:
        # Consistent string representation of missing and present categories.
        data[col] = data[col].astype('string').fillna('__missing__').astype(str)

    (train_pool, valid_rows, test_rows), split_boundaries = split_chronologically(data)
    if max_rows is not None and max_rows < 100:
        raise ValueError('max_rows must be at least 100')
    if not np.isfinite(positive_weight_multiplier) or positive_weight_multiplier <= 0:
        raise ValueError('positive_weight_multiplier must be a finite positive number')
    if mode == 'final' and positive_weight_multiplier != 1.0:
        raise ValueError('positive weight candidates are validation-only; final mode requires multiplier 1.0')
    train_rows = train_pool.tail(max_rows) if max_rows is not None else train_pool
    if train_rows['isFraud'].nunique() != 2:
        raise ValueError('Selected train rows must contain both classes')
    if FORBIDDEN.intersection(features):
        raise ValueError('Forbidden target, identifier, or split column in features')
    del data
    experiment, started = start_experiment(
        dataset='IEEE-CIS', algorithm='XGBoost', features=features,
        split_method='chronological TransactionDT 70/15/15; final test omitted in experiment mode', mode=mode,
        hyperparameters={'n_estimators': n_estimators, 'max_train_rows': max_rows,
                         'max_false_alerts_per_10000': max_false_alerts_per_10000,
                         'max_depth': 4, 'learning_rate': .065, 'feature_set': feature_set,
                         'positive_weight_multiplier': positive_weight_multiplier},
    )

    transform = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', add_indicator=True), nums),
        ('category', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=100,
                                   max_categories=40), cats),
    ], sparse_threshold=.3)
    positives = train_rows['isFraud'].sum()
    negatives = len(train_rows) - positives
    classifier = XGBClassifier(
        n_estimators=n_estimators, max_depth=4, learning_rate=.065, subsample=.85,
        colsample_bytree=.85, reg_lambda=3, objective='binary:logistic',
        tree_method='hist', eval_metric='aucpr', random_state=42,
        n_jobs=min(4, os.cpu_count() or 1), scale_pos_weight=float(negatives / positives) * positive_weight_multiplier,
    )
    pipeline = Pipeline([('preprocessing', transform), ('model', classifier)])
    x_train, y_train = train_rows[features], train_rows['isFraud'].astype(int)
    baseline_transform = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scale', StandardScaler(with_mean=False)),
                              ('clip', FunctionTransformer(np.clip, kw_args={'a_min': -20, 'a_max': 20}))]), nums),
        ('category', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=100, max_categories=40), cats),
    ], sparse_threshold=.3)
    baseline = Pipeline([('preprocessing', baseline_transform),
                         ('model', LogisticRegression(max_iter=1000, tol=1e-3, class_weight='balanced', solver='liblinear'))])
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=ConvergenceWarning)
        baseline.fit(x_train, y_train)
    if baseline.named_steps['model'].n_iter_[0] >= 1000:
        raise RuntimeError('Baseline logistic regression did not converge')
    pipeline.fit(x_train, y_train)
    validation_scores = pipeline.predict_proba(valid_rows[features])[:, 1]
    threshold, analysis = threshold_analysis(valid_rows['isFraud'].astype(int), validation_scores, max_false_alerts_per_10000)
    validation_metrics = metrics(valid_rows['isFraud'].astype(int), validation_scores, threshold)
    baseline_validation_scores = baseline.predict_proba(valid_rows[features])[:, 1]
    baseline_validation_metrics = metrics(valid_rows['isFraud'].astype(int), baseline_validation_scores, .5)
    importance = feature_importance(pipeline, features)
    test_metrics = None
    baseline_test_metrics = None
    if mode == 'final':
        test_scores = pipeline.predict_proba(test_rows[features])[:, 1]
        test_metrics = metrics(test_rows['isFraud'].astype(int), test_scores, threshold)
        baseline_scores = baseline.predict_proba(test_rows[features])[:, 1]
        baseline_test_metrics = metrics(test_rows['isFraud'].astype(int), baseline_scores, .5)

    artifact = {
        'format_version': 2,
        'model_kind': 'ieee_cis',
        'model_version': 'ieee-cis-xgboost-v2',
        'run_mode': mode,
        'feature_set': feature_set,
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'IEEE-CIS Fraud Detection (Kaggle competition)',
        'data_files': [file_provenance(tx_path), file_provenance(id_path)],
        'library_versions': {'python': sys.version.split()[0], 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__},
        'split_counts': {'train_pool': len(train_pool), 'train_used': len(train_rows), 'validation': len(valid_rows), 'final_test': len(test_rows)},
        'split_boundaries': split_boundaries,
        'experiment': finish_experiment(experiment, started),
        'pipeline': pipeline,
        'features': features,
        'numerical_features': nums,
        'categorical_features': cats,
        'threshold': threshold,
        'validation_metrics': validation_metrics,
        'baseline_validation_metrics': baseline_validation_metrics,
        'baseline_convergence': {'solver': 'liblinear', 'iterations': int(baseline.named_steps['model'].n_iter_[0]), 'max_iter': 1000, 'converged': True,
                                 'numeric_preprocessing': 'train-fitted standard scaling and clipping to [-20,20]'},
        'threshold_analysis': analysis,
        'validation_ranking_metrics': ranking_metrics(valid_rows['isFraud'].astype(int), validation_scores),
        'validation_operating_profiles': operating_profiles(valid_rows['isFraud'].astype(int), validation_scores, threshold),
        'feature_importance': importance,
        'test_metrics': test_metrics,
        'baseline_test_metrics': baseline_test_metrics,
        'split_method': 'fixed full-label 70/15/15 chronological TransactionDT; train subset only within first 70%',
        'warning': 'Research dataset; scores not calibrated for deployment and not a guarantee of banking fraud detection.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'final':
        joblib.dump(artifact, output, compress=3)
    report = {key: value for key, value in artifact.items() if key != 'pipeline'}
    report['threshold'] = round(threshold, 6)
    report_path = output.with_suffix('.metrics.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'model_path': str(output) if mode == 'final' else None, 'metrics_path': str(report_path),
                      'validation': validation_metrics, 'test': test_metrics, 'baseline_test': baseline_test_metrics}, ensure_ascii=False, indent=2))
    if mode == 'final':
        print('IMPORTANT: The fixed held-out test set was evaluated once after all choices on validation.')
    return report


def main():
    parser = argparse.ArgumentParser(description='Train IEEE-CIS ML model on chronological splits')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('models/ieee_model.joblib'))
    parser.add_argument('--estimators', type=int, default=180)
    parser.add_argument('--mode', choices=['experiment', 'final'], default='experiment')
    parser.add_argument('--max-train-rows', type=int, default=None, help='Use most recent N rows inside the fixed train window')
    parser.add_argument('--max-false-alerts-per-10000', type=float, default=None, help='Validation constraint; otherwise maximize F1')
    parser.add_argument('--feature-set', choices=['core', 'extended_cv'], default='core',
                        help='Candidate extended C/V features; evaluate in experiment mode before any final use')
    parser.add_argument('--positive-weight-multiplier', type=float, default=1.0,
                        help='Experiment-only multiplier over inverse-prevalence class weight')
    args = parser.parse_args()
    if args.max_train_rows is not None and args.max_train_rows < 100:
        parser.error('--max-train-rows must be at least 100')
    if args.mode == 'experiment' and args.output.name == 'ieee_model.joblib':
        parser.error('Experiments require a distinct --output path under models/experiments/')
    if args.mode == 'final' and args.max_train_rows is not None:
        parser.error('Final run uses all rows in the fixed train window')
    train(args.data_dir, args.output, args.estimators, args.max_train_rows, args.mode,
          args.max_false_alerts_per_10000, args.feature_set, args.positive_weight_multiplier)

if __name__ == '__main__':
    main()
