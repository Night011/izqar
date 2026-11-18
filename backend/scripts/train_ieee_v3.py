"""Train an improved IEEE-CIS transaction classifier (experimental v3 candidate).

Improvements over train_ieee.py (ieee_final_v2):
  * Full C/D/M/V feature set instead of V1..V3 only.
  * Engineered amount/time/email/missingness features.
  * Train-fold frequency encodings for high-cardinality identifiers (no leakage).
  * Native XGBoost NaN handling for numerics; early stopping on validation aucpr.
  * Extended metrics: Precision@100/@1000, false alerts per 10k, FN count.

Protocol: identical chronological 70/15/15 TransactionDT split; threshold tuned
on validation only; test evaluated once. Never trains on synthetic simulators.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

BASE_NUMERIC = [
    'TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2',
    'dist1', 'dist2',
] + [f'C{i}' for i in range(1, 15)] + [f'D{i}' for i in range(1, 16)] + [f'V{i}' for i in range(1, 340)]
CATEGORICAL = [
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'DeviceType', 'DeviceInfo', 'id_31',
]
ENGINEERED_NUMERIC = ['amt_log', 'amt_cents', 'dt_hour', 'dt_day', 'email_match',
                      'v_missing', 'd_missing',
                      'card1_freq', 'card1_card5_freq', 'card2_freq', 'addr1_freq',
                      'p_email_freq', 'device_freq']
FREQ_SOURCES = {'card1_freq': ['card1'], 'card1_card5_freq': ['card1', 'card5'],
                'card2_freq': ['card2'], 'addr1_freq': ['addr1'],
                'p_email_freq': ['P_emaildomain'], 'device_freq': ['DeviceInfo']}


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    amt = pd.to_numeric(df['TransactionAmt'], errors='coerce')
    df['amt_log'] = np.log1p(amt)
    df['amt_cents'] = (amt * 100) % 100
    dt = pd.to_numeric(df['TransactionDT'], errors='coerce')
    df['dt_hour'] = (dt // 3600) % 24
    df['dt_day'] = dt // 86400
    p, r = df.get('P_emaildomain'), df.get('R_emaildomain')
    df['email_match'] = ((p.astype('string') == r.astype('string')) & p.notna() & r.notna()).astype('float32')
    vcols = [c for c in df.columns if c.startswith('V') and c[1:].isdigit()]
    dcols = [f'D{i}' for i in range(1, 16)]
    df['v_missing'] = df[vcols].isna().sum(axis=1).astype('float32')
    df['d_missing'] = df[dcols].isna().sum(axis=1).astype('float32')
    return df


def fit_freq_maps(train: pd.DataFrame) -> dict[str, dict]:
    maps = {}
    for name, cols in FREQ_SOURCES.items():
        key = train[cols].astype('string').fillna('__missing__').agg('_'.join, axis=1) if len(cols) > 1 \
            else train[cols[0]].astype('string').fillna('__missing__')
        maps[name] = key.value_counts(normalize=True).to_dict()
    return maps


def apply_freq_maps(df: pd.DataFrame, maps: dict[str, dict]) -> pd.DataFrame:
    df = df.copy()
    for name, cols in FREQ_SOURCES.items():
        key = df[cols].astype('string').fillna('__missing__').agg('_'.join, axis=1) if len(cols) > 1 \
            else df[cols[0]].astype('string').fillna('__missing__')
        df[name] = key.map(maps[name]).fillna(0).astype('float32')
    return df


def metrics(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    order = np.argsort(-scores)
    y_sorted = np.asarray(y_true)[order]
    p_at = lambda k: float(y_sorted[:k].mean()) if len(y_sorted) >= k else float(y_sorted.mean())
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'average_precision_pr_auc': round(float(average_precision_score(y_true, scores)), 6),
        'roc_auc': round(float(roc_auc_score(y_true, scores)), 6),
        'precision': round(float(precision_score(y_true, pred, zero_division=0)), 6),
        'recall': round(float(recall_score(y_true, pred, zero_division=0)), 6),
        'f1': round(float(f1_score(y_true, pred, zero_division=0)), 6),
        'precision_at_100': round(p_at(100), 6),
        'precision_at_1000': round(p_at(1000), 6),
        'false_positive_rate': round(float(fp / max(fp + tn, 1)), 6),
        'false_negative_rate': round(float(fn / max(fn + tp, 1)), 6),
        'false_alerts_per_10000_operations': round(float(fp / max(len(y_true), 1) * 10000), 3),
        'false_negatives': int(fn),
        'confusion_matrix_tn_fp_fn_tp': [int(tn), int(fp), int(fn), int(tp)],
        'cases': int(len(y_true)),
        'fraud_cases': int(np.sum(y_true)),
    }


def choose_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def train(data_dir: Path, output: Path, n_estimators: int, max_rows: int | None, seed: int):
    t0 = time.time()
    tx_path = data_dir / 'train_transaction.csv'
    id_path = data_dir / 'train_identity.csv'
    if not tx_path.exists():
        raise FileNotFoundError(f'Missing {tx_path}. Download IEEE-CIS from Kaggle.')
    data = pd.read_csv(tx_path, nrows=max_rows, low_memory=False)
    if id_path.exists():
        identity = pd.read_csv(id_path, low_memory=False)
        data = data.merge(identity, on='TransactionID', how='left', validate='one_to_one')
    data = data.sort_values(['TransactionDT', 'TransactionID'], kind='stable').reset_index(drop=True)
    data = add_engineered(data)

    first, second = int(len(data) * .70), int(len(data) * .85)
    train_rows, valid_rows, test_rows = data.iloc[:first], data.iloc[first:second], data.iloc[second:]
    for label, part in [('train', train_rows), ('validation', valid_rows), ('test', test_rows)]:
        if part['isFraud'].nunique() != 2:
            raise ValueError(f'{label} split must contain both classes.')

    freq_maps = fit_freq_maps(train_rows)
    train_rows = apply_freq_maps(train_rows, freq_maps)
    valid_rows = apply_freq_maps(valid_rows, freq_maps)
    test_rows = apply_freq_maps(test_rows, freq_maps)

    nums = [c for c in BASE_NUMERIC + ENGINEERED_NUMERIC if c in data.columns]
    cats = [c for c in CATEGORICAL if c in data.columns]
    features = nums + cats
    for col in nums:
        train_rows[col] = pd.to_numeric(train_rows[col], errors='coerce').astype('float32')
        valid_rows[col] = pd.to_numeric(valid_rows[col], errors='coerce').astype('float32')
        test_rows[col] = pd.to_numeric(test_rows[col], errors='coerce').astype('float32')
    for col in cats:
        for part in (train_rows, valid_rows, test_rows):
            part[col] = part[col].astype('string').fillna('__missing__').astype(str)

    transform = ColumnTransformer([
        ('numeric', 'passthrough', nums),
        ('category', OneHotEncoder(handle_unknown='ignore', min_frequency=100,
                                   max_categories=60), cats),
    ], sparse_threshold=.3)
    positives = int(train_rows['isFraud'].sum())
    negatives = len(train_rows) - positives
    classifier = XGBClassifier(
        n_estimators=n_estimators, max_depth=5, learning_rate=.05, subsample=.8,
        colsample_bytree=.6, reg_lambda=4, objective='binary:logistic',
        tree_method='hist', eval_metric='aucpr', random_state=seed,
        n_jobs=min(8, os.cpu_count() or 1), scale_pos_weight=float(negatives / positives),
        early_stopping_rounds=60,
    )
    x_train, y_train = train_rows[features], train_rows['isFraud'].astype(int)
    x_valid, y_valid = valid_rows[features], valid_rows['isFraud'].astype(int)
    x_test, y_test = test_rows[features], test_rows['isFraud'].astype(int)
    # Fit the transformer once so the eval set can be passed to XGBoost early stopping.
    x_train_t = transform.fit_transform(x_train)
    x_valid_t = transform.transform(x_valid)
    x_test_t = transform.transform(x_test)
    classifier.fit(x_train_t, y_train, eval_set=[(x_valid_t, y_valid)], verbose=False)
    best_iteration = getattr(classifier, 'best_iteration', None)
    pipeline = Pipeline([('preprocessing', transform), ('model', classifier)])

    validation_scores = pipeline.predict_proba(x_valid)[:, 1]
    threshold = choose_threshold(y_valid.to_numpy(), validation_scores)
    test_scores = pipeline.predict_proba(x_test)[:, 1]
    validation_metrics = metrics(y_valid.to_numpy(), validation_scores, threshold)
    test_metrics = metrics(y_test.to_numpy(), test_scores, threshold)
    train_seconds = round(time.time() - t0, 1)

    artifact = {
        'format_version': 1,
        'model_version': 'ieee-cis-xgboost-v3-candidate',
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'IEEE-CIS Fraud Detection (Kaggle competition)',
        'pipeline': pipeline,
        'features': features,
        'numerical_features': nums,
        'categorical_features': cats,
        'threshold': threshold,
        'validation_metrics': validation_metrics,
        'test_metrics': test_metrics,
        'split_method': '70/15/15 chronological TransactionDT ascending (identical to ieee_final_v2)',
        'best_iteration': int(best_iteration) if best_iteration is not None else None,
        'train_seconds': train_seconds,
        'warning': 'Research dataset; scores not calibrated for deployment and not a guarantee of banking fraud detection.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    report = {k: v for k, v in artifact.items() if k != 'pipeline'}
    report['threshold'] = round(threshold, 6)
    output.with_suffix('.metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'model_path': str(output), 'best_iteration': artifact['best_iteration'],
                      'train_seconds': train_seconds,
                      'validation': validation_metrics, 'test': test_metrics}, ensure_ascii=False, indent=2))
    print('IMPORTANT: The held-out test set was evaluated once after threshold selection on validation.')


def main():
    parser = argparse.ArgumentParser(description='Train improved IEEE-CIS v3 candidate')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('models/experiments/ieee_v3_candidate.joblib'))
    parser.add_argument('--estimators', type=int, default=1500)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args.data_dir, args.output, args.estimators, args.max_rows, args.seed)


if __name__ == '__main__':
    main()
