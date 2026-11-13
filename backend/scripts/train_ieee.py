"""Train a reproducible IEEE-CIS transaction classifier with a chronological holdout.

This model NEVER trains on the synthetic transfer simulator.
It doesn't imply real-world fraud performance or calibrated probabilities.
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

NUMERIC = [
    'TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2',
    'dist1', 'dist2', 'C1', 'C2', 'C4', 'C6', 'C7', 'C8', 'C9', 'C10',
    'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D15',
    'V1', 'V2', 'V3',
]
CATEGORICAL = [
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'DeviceType', 'DeviceInfo', 'id_31',
]


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


def train(data_dir: Path, output: Path, n_estimators: int, max_rows: int | None):
    tx_path = data_dir / 'train_transaction.csv'
    id_path = data_dir / 'train_identity.csv'
    if not tx_path.exists():
        raise FileNotFoundError(f'Missing {tx_path}. Download IEEE-CIS from Kaggle.')
    tx_columns = set(pd.read_csv(tx_path, nrows=0).columns)
    required = {'TransactionID', 'TransactionDT', 'TransactionAmt', 'isFraud'}
    if missing := (required - tx_columns):
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    tx_usecols = sorted((tx_columns & set(NUMERIC + CATEGORICAL)) | required)
    data = pd.read_csv(tx_path, usecols=tx_usecols, nrows=max_rows, low_memory=False)

    if id_path.exists():
        identity_cols = set(pd.read_csv(id_path, nrows=0).columns)
        usecols = sorted(identity_cols & (set(CATEGORICAL) | {'TransactionID'}))
        if len(usecols) > 1:
            identity = pd.read_csv(id_path, usecols=usecols, low_memory=False)
            data = data.merge(identity, on='TransactionID', how='left', validate='one_to_one')
    data = data.sort_values(['TransactionDT', 'TransactionID'], kind='stable').reset_index(drop=True)
    nums = [column for column in NUMERIC if column in data]
    cats = [column for column in CATEGORICAL if column in data]
    features = nums + cats
    for col in nums:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    for col in cats:
        # Consistent string representation of missing and present categories.
        data[col] = data[col].astype('string').fillna('__missing__').astype(str)

    if len(data) < 100:
        raise ValueError('Need at least 100 rows for chronological training and evaluation')
    first, second = int(len(data) * .70), int(len(data) * .85)
    train_rows, valid_rows, test_rows = data.iloc[:first], data.iloc[first:second], data.iloc[second:]
    for label, part in [('train', train_rows), ('validation', valid_rows), ('test', test_rows)]:
        if part['isFraud'].nunique() != 2:
            raise ValueError(f'{label} split must contain both classes. Select more chronological rows.')

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
        n_jobs=min(4, os.cpu_count() or 1), scale_pos_weight=float(negatives / positives),
    )
    pipeline = Pipeline([('preprocessing', transform), ('model', classifier)])
    x_train, y_train = train_rows[features], train_rows['isFraud'].astype(int)
    # A simple baseline makes the XGBoost result interpretable; it is not used for threshold selection.
    baseline = Pipeline([('preprocessing', transform), ('model', LogisticRegression(max_iter=250, class_weight='balanced', solver='saga', n_jobs=min(4, os.cpu_count() or 1)))])
    baseline.fit(x_train, y_train)
    pipeline.fit(x_train, y_train)
    validation_scores = pipeline.predict_proba(valid_rows[features])[:, 1]
    threshold = choose_threshold(valid_rows['isFraud'].astype(int), validation_scores)
    test_scores = pipeline.predict_proba(test_rows[features])[:, 1]
    validation_metrics = metrics(valid_rows['isFraud'].astype(int), validation_scores, threshold)
    test_metrics = metrics(test_rows['isFraud'].astype(int), test_scores, threshold)
    baseline_scores = baseline.predict_proba(test_rows[features])[:, 1]
    baseline_test_metrics = metrics(test_rows['isFraud'].astype(int), baseline_scores, .5)

    artifact = {
        'format_version': 2,
        'model_kind': 'ieee_cis',
        'model_version': 'ieee-cis-xgboost-v1',
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'IEEE-CIS Fraud Detection (Kaggle competition)',
        'pipeline': pipeline,
        'features': features,
        'numerical_features': nums,
        'categorical_features': cats,
        'threshold': threshold,
        'validation_metrics': validation_metrics,
        'test_metrics': test_metrics,
        'baseline_test_metrics': baseline_test_metrics,
        'split_method': '70/15/15 chronological TransactionDT ascending',
        'warning': 'Research dataset; scores not calibrated for deployment and not a guarantee of banking fraud detection.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    report = {key: value for key, value in artifact.items() if key != 'pipeline'}
    report['threshold'] = round(threshold, 6)
    report_path = output.with_suffix('.metrics.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'model_path': str(output), 'metrics_path': str(report_path),
                      'validation': validation_metrics, 'test': test_metrics, 'baseline_test': baseline_test_metrics}, ensure_ascii=False, indent=2))
    print('IMPORTANT: The held-out test set was evaluated once after threshold selection on validation.')


def main():
    parser = argparse.ArgumentParser(description='Train IEEE-CIS ML model on chronological splits')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('models/ieee_model.joblib'))
    parser.add_argument('--estimators', type=int, default=180)
    parser.add_argument('--max-rows', type=int, default=None)
    args = parser.parse_args()
    train(args.data_dir, args.output, args.estimators, args.max_rows)

if __name__ == '__main__':
    main()
