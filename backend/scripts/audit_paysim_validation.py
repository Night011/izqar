"""Validation-only PaySim robustness audit for a frozen model artifact."""
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
from train_paysim import evaluate, read_paysim


def _segment(y: np.ndarray, scores: np.ndarray, indices: np.ndarray, threshold: float) -> dict:
    if not len(indices):
        return {'cases': 0}
    labels, values = y[indices], scores[indices]
    positive = values >= threshold
    tp = int(np.sum(positive & (labels == 1)))
    fp = int(np.sum(positive & (labels == 0)))
    fn = int(np.sum((~positive) & (labels == 1)))
    return {'cases': len(indices), 'fraud_cases': int(labels.sum()),
            'precision': round(tp / max(tp + fp, 1), 6),
            'recall': round(tp / max(tp + fn, 1), 6),
            'false_alerts_per_10000_operations': round(fp * 10000 / len(indices), 3),
            'pr_auc': round(float(average_precision_score(labels, values)), 6) if len(np.unique(labels)) == 2 else None}


def audit(data_path: Path, model_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f'Report already exists: {output}')
    artifact = _load_artifact(model_path, 'paysim')
    if artifact is None:
        raise FileNotFoundError(model_path)
    data = read_paysim(data_path, max_rows=None)
    first = data.step.iloc[int(len(data) * .70)]
    second = data.step.iloc[int(len(data) * .85)]
    validation = data[(data.step >= first) & (data.step < second)].copy()
    del data
    if len(validation) != artifact['split_counts']['validation']:
        raise ValueError('Validation row count differs from the frozen model split')
    x = validation[artifact['features']]
    y = validation.isFraud.to_numpy(dtype=np.int8)
    scores = artifact['pipeline'].predict_proba(x)[:, 1]
    threshold = float(artifact['threshold'])
    time_indices = np.array_split(np.arange(len(validation)), 3)
    amount_labels = pd.qcut(validation.amount, q=4, duplicates='drop').astype(str).to_numpy()
    types = validation.type.astype(str).to_numpy()
    k = min(100, len(y))
    top_k = np.argsort(-scores, kind='stable')[:k]
    report = {
        'analysis_source': 'frozen_paysim_validation_only',
        'model_version': artifact['model_version'],
        'model_trained_at_utc': artifact.get('trained_at_utc'),
        'step_boundaries': {'validation_start': int(first), 'test_start': int(second)},
        'cases': len(validation), 'threshold': threshold,
        'overall': evaluate(y, scores, threshold),
        'precision_at_100': round(float(y[top_k].mean()), 6),
        'recall_at_100': round(float(y[top_k].sum() / max(y.sum(), 1)), 6),
        'brier_score_uncalibrated': round(float(brier_score_loss(y, scores)), 6),
        'time_thirds': {name: _segment(y, scores, indices, threshold)
                        for name, indices in zip(('early', 'middle', 'late'), time_indices)},
        'amount_quartiles': {name: _segment(y, scores, np.flatnonzero(amount_labels == name), threshold)
                             for name in sorted(set(amount_labels))},
        'transaction_types': {name: _segment(y, scores, np.flatnonzero(types == name), threshold)
                              for name in sorted(set(types))},
        'stress_checks': {},
        'limitations': 'PaySim is synthetic and validation selected the threshold. Stress tests are diagnostic, not independent generalization estimates.',
    }
    changed_input = x.copy()
    changed_input['type'] = '__unseen_type__'
    changed = artifact['pipeline'].predict_proba(changed_input)[:, 1]
    report['stress_checks']['unseen_transaction_type'] = {
        'pr_auc': round(float(average_precision_score(y, changed)), 6),
        'mean_absolute_score_change': round(float(np.mean(np.abs(changed - scores))), 6)}
    changed_input = x.copy()
    rng = np.random.default_rng(42)
    rows = rng.choice(len(x), size=max(1, len(x) // 10), replace=False)
    changed_input.loc[validation.index[rows], artifact['numerical_features']] = np.nan
    changed = artifact['pipeline'].predict_proba(changed_input)[:, 1]
    report['stress_checks']['numeric_missing_10pct_rows'] = {
        'pr_auc': round(float(average_precision_score(y, changed)), 6),
        'mean_absolute_score_change': round(float(np.mean(np.abs(changed - scores))), 6)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='Audit PaySim validation without scoring the test window')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data, args.model, args.output)
    print(json.dumps({'report': str(args.output), 'cases': result['cases'], 'overall': result['overall']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
