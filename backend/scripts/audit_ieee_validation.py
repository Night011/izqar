"""Audit the frozen IEEE-CIS model on its validation window only.

No final-test labels or scores are used for model selection. Input CSVs remain
unchanged; the output is a separate validation-only report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml import _load_artifact
from train_ieee import metrics


def load_validation(data_dir: Path, artifact: dict) -> pd.DataFrame:
    tx_path = data_dir / 'train_transaction.csv'
    id_path = data_dir / 'train_identity.csv'
    if not tx_path.is_file() or not id_path.is_file():
        raise FileNotFoundError('Both IEEE-CIS labeled train CSVs are required')
    start = artifact['split_boundaries']['validation_start_dt']
    stop = artifact['split_boundaries']['test_start_dt']
    features = artifact['features']
    header = set(pd.read_csv(tx_path, nrows=0).columns)
    columns = set(features) & header
    required = {'TransactionID', 'TransactionDT', 'isFraud'}
    if not required <= header:
        raise ValueError(f'Missing transaction columns: {sorted(required - header)}')
    dtypes = {name: 'float32' for name in artifact['numerical_features'] if name in columns}
    dtypes.update({'TransactionID': 'int64', 'TransactionDT': 'float64', 'isFraud': 'int8'})
    selected = []
    for chunk in pd.read_csv(tx_path, usecols=sorted(columns | required), dtype=dtypes, chunksize=100_000):
        part = chunk[(chunk.TransactionDT >= start) & (chunk.TransactionDT < stop)]
        if not part.empty:
            selected.append(part)
    if not selected:
        raise ValueError('No rows in the frozen validation window')
    validation = pd.concat(selected, ignore_index=True)
    if not validation.TransactionID.is_unique:
        raise ValueError('Duplicate TransactionID in validation transactions')
    expected = artifact['split_counts']['validation']
    if len(validation) != expected:
        raise ValueError(f'Validation window changed: expected {expected} rows, found {len(validation)}')

    identity_header = set(pd.read_csv(id_path, nrows=0).columns)
    identity_columns = (set(features) & identity_header) | {'TransactionID'}
    if 'TransactionID' not in identity_header:
        raise ValueError('Identity CSV lacks TransactionID')
    wanted = set(validation.TransactionID)
    identity_parts = []
    for chunk in pd.read_csv(id_path, usecols=sorted(identity_columns), chunksize=100_000):
        part = chunk[chunk.TransactionID.isin(wanted)]
        if not part.empty:
            identity_parts.append(part)
    if identity_parts:
        identity = pd.concat(identity_parts, ignore_index=True)
        if not identity.TransactionID.is_unique:
            raise ValueError('Duplicate TransactionID in validation identity rows')
        validation = validation.merge(identity, on='TransactionID', how='left', validate='one_to_one')
    for name in features:
        if name not in validation:
            validation[name] = pd.NA
    for name in artifact['numerical_features']:
        validation[name] = pd.to_numeric(validation[name], errors='coerce').astype('float32')
    for name in artifact['categorical_features']:
        validation[name] = validation[name].astype('string').fillna('__missing__').astype(str)
    return validation.sort_values(['TransactionDT', 'TransactionID'], kind='stable').reset_index(drop=True)


def _segment(y: np.ndarray, scores: np.ndarray, indices: np.ndarray, threshold: float) -> dict:
    if not len(indices):
        return {'cases': 0}
    labels = y[indices]
    values = scores[indices]
    prediction = values >= threshold
    tp = int(np.sum(prediction & (labels == 1)))
    fp = int(np.sum(prediction & (labels == 0)))
    fn = int(np.sum((~prediction) & (labels == 1)))
    return {'cases': len(indices), 'fraud_cases': int(labels.sum()),
            'precision': round(tp / max(tp + fp, 1), 6),
            'recall': round(tp / max(tp + fn, 1), 6),
            'false_alerts_per_10000_operations': round(fp * 10000 / len(indices), 3),
            'pr_auc': round(float(average_precision_score(labels, values)), 6) if len(np.unique(labels)) == 2 else None,
            'score_p50': round(float(np.median(values)), 6), 'score_p99': round(float(np.quantile(values, .99)), 6)}


def audit(data_dir: Path, model_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f'Report already exists: {output}')
    artifact = _load_artifact(model_path, 'ieee_cis')
    if artifact is None:
        raise FileNotFoundError(model_path)
    if artifact.get('run_mode') != 'final' or artifact.get('test_metrics') is None:
        raise ValueError('A frozen final IEEE-CIS artifact is required')
    validation = load_validation(data_dir, artifact)
    features = artifact['features']
    x = validation[features]
    y = validation.isFraud.to_numpy(dtype=np.int8)
    scores = artifact['pipeline'].predict_proba(x)[:, 1]
    threshold = float(artifact['threshold'])
    time_indices = np.array_split(np.arange(len(validation)), 3)
    amount = validation.TransactionAmt
    amount_groups = pd.qcut(amount, q=4, duplicates='drop')
    amount_labels = amount_groups.astype(str).to_numpy()
    product = validation.ProductCD.to_numpy() if 'ProductCD' in validation else np.full(len(y), '__absent__')
    top_products = pd.Series(product).value_counts().head(5).index
    k = min(100, len(y))
    top_k = np.argsort(-scores, kind='stable')[:k]
    report = {
        'analysis_source': 'frozen_ieee_validation_only',
        'model_version': artifact['model_version'],
        'model_trained_at_utc': artifact.get('trained_at_utc'),
        'split_boundaries': artifact['split_boundaries'],
        'cases': len(validation),
        'threshold': threshold,
        'overall': metrics(y, scores, threshold),
        'precision_at_100': round(float(y[top_k].mean()), 6),
        'recall_at_100': round(float(y[top_k].sum() / max(y.sum(), 1)), 6),
        'brier_score_uncalibrated': round(float(brier_score_loss(y, scores)), 6),
        'time_thirds': {name: _segment(y, scores, indices, threshold) for name, indices in zip(('early', 'middle', 'late'), time_indices)},
        'amount_quartiles': {name: _segment(y, scores, np.flatnonzero(amount_labels == name), threshold)
                             for name in sorted(set(amount_labels))},
        'product_types': {str(name): _segment(y, scores, np.flatnonzero(product == name), threshold)
                          for name in top_products},
        'stress_checks': {},
        'limitations': 'Validation was used to choose the frozen threshold and may already have guided model development. These are diagnostic, not independent final-test metrics.',
    }
    if 'DeviceInfo' in features:
        altered = x.copy()
        altered['DeviceInfo'] = '__unseen_device__'
        changed = artifact['pipeline'].predict_proba(altered)[:, 1]
        report['stress_checks']['unseen_device_category'] = {
            'pr_auc': round(float(average_precision_score(y, changed)), 6),
            'mean_absolute_score_change': round(float(np.mean(np.abs(changed - scores))), 6)}
    altered = x.copy()
    numeric = artifact['numerical_features']
    rng = np.random.default_rng(42)
    rows = rng.choice(len(altered), size=max(1, len(altered) // 10), replace=False)
    altered.loc[rows, numeric] = np.nan
    changed = artifact['pipeline'].predict_proba(altered)[:, 1]
    report['stress_checks']['numeric_missing_10pct_rows'] = {
        'pr_auc': round(float(average_precision_score(y, changed)), 6),
        'mean_absolute_score_change': round(float(np.mean(np.abs(changed - scores))), 6)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='Audit IEEE-CIS validation without touching final test')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_dir, args.model, args.output)
    print(json.dumps({'report': str(args.output), 'cases': result['cases'], 'overall': result['overall']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
