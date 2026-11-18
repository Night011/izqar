"""Reproduce the official Feedzai BAF benchmark, separately from our temporal one.

Official protocol (notebooks/empirical_results.ipynb in feedzai/bank-account-fraud):
months 0-5 train, months 6-7 test, every column except the label used as a
feature, category columns as LightGBM categoricals, hyperparameters sampled from
notebooks/lightgbm_hyperparameter_space.yaml, and the operating point set at the
lowest threshold whose false-positive rate is still below 5% (about 500 FP per
10k) with recall reported.

This is a *historical* reference, not independent confirmation: months 6-7 were
inspected earlier in the project, and the official operating point is fitted on
the test labels. The point of this script is (a) to check the magnitude of the
published benchmark and (b) to measure what the official feature contract adds
over the admissible one.

Feature variants:
  official    every column but fraud_bool (what the notebook does)
  no_month    minus month (a time index cannot be known at scoring time)
  admissible  minus month, days_since_request, device_fraud_count,
              credit_risk_score - the contract used by the research folds

Deviations from the notebook, all recorded in the output: 12 random trials
instead of 100, n_estimators capped at 600 (the published space allows 10000),
and pandas category dtype instead of LabelEncoder so unseen levels do not raise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baf import CATEGORICAL, VELOCITY, NUMERIC  # noqa: E402

POST_HOC_BUDGETS = (50, 100, 200, 500)


def official_space(trial, rng):
    """Sample from lightgbm_hyperparameter_space.yaml, with the documented caps."""
    return {
        'n_estimators': int(np.exp(rng.uniform(np.log(20), np.log(600)))),
        'max_depth': int(rng.integers(3, 13)),
        'learning_rate': float(np.exp(rng.uniform(np.log(0.02), np.log(0.1)))),
        'num_leaves': int(np.exp(rng.uniform(np.log(10), np.log(100)))),
        'boosting_type': str(rng.choice(['gbdt', 'goss'])),
        'min_data_in_leaf': int(np.exp(rng.uniform(np.log(5), np.log(200)))),
        'max_bin': int(rng.integers(100, 501)),
        'enable_bundle': bool(rng.integers(0, 2)),
    }


def recall_at_budget(y_true, scores, fp_per_10k):
    """Largest recall among thresholds whose false-alert rate fits the budget."""
    y_true = np.asarray(y_true)
    negatives = int((y_true == 0).sum())
    positives = int((y_true == 1).sum())
    allowed = int(np.floor(fp_per_10k * negatives / 10000))
    order = np.argsort(-np.asarray(scores), kind='stable')
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    feasible = fp <= allowed
    if not feasible.any():
        return 0.0, float('inf')
    best = int(np.max(tp[feasible]))
    cutoff = float(np.asarray(scores)[order][np.max(np.nonzero(feasible)[0])])
    return round(best / max(positives, 1), 6), cutoff


def operate_at_fpr(y_true, scores, target_fpr=0.05):
    """Official rule: lowest threshold whose FPR is still under the target."""
    from sklearn.metrics import roc_curve
    fprs, tprs, thresholds = roc_curve(y_true, scores)
    mask = fprs < target_fpr
    index = int(np.max(np.nonzero(mask)[0]))
    return float(thresholds[index]), float(tprs[index])


def evaluate(y_true, scores):
    metrics = {'average_precision_pr_auc': round(float(average_precision_score(y_true, scores)), 6),
               'roc_auc': round(float(roc_auc_score(y_true, scores)), 6)}
    threshold, recall = operate_at_fpr(y_true, scores)
    predicted = np.asarray(scores) >= threshold
    fp = int(np.sum(predicted & (np.asarray(y_true) == 0)))
    metrics['official_operating_point'] = {
        'threshold': round(threshold, 6), 'recall': round(recall, 6),
        'false_alerts_per_10000': round(fp * 10000 / max(int((np.asarray(y_true) == 0).sum()), 1), 3),
        'note': 'threshold fitted on these labels by the published protocol'}
    for budget in POST_HOC_BUDGETS:
        value, cutoff = recall_at_budget(y_true, scores, budget)
        metrics[f'recall_at_{budget}_fp_10k'] = value
        metrics[f'threshold_at_{budget}_fp_10k'] = round(cutoff, 6)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('models/experiments/baf_official_repro.json'))
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    from lightgbm import LGBMClassifier

    frame = pd.read_csv(args.data)
    for name in CATEGORICAL:
        frame[name] = frame[name].astype('category')
    train = frame[frame.month < 6]
    test = frame[frame.month >= 6]
    y_train = train.fraud_bool.to_numpy()
    y_test = test.fraud_bool.to_numpy()
    positive = int(y_train.sum())
    print(f'train {len(train)} rows ({positive} frauds), test {len(test)} rows '
          f'({int(y_test.sum())} frauds)', flush=True)

    variants = {
        'official': [c for c in frame.columns if c != 'fraud_bool'],
        'no_month': [c for c in frame.columns if c not in {'fraud_bool', 'month'}],
        'admissible': list(NUMERIC) + list(VELOCITY) + list(CATEGORICAL),
    }

    rng = np.random.default_rng(args.seed)
    trials = []
    for index in range(args.trials):
        params = official_space(index, rng)
        model = LGBMClassifier(**params, n_jobs=4, verbose=-1, random_state=args.seed)
        model.fit(train[variants['official']], y_train, categorical_feature=CATEGORICAL)
        scores = model.predict_proba(test[variants['official']])[:, 1]
        metrics = evaluate(y_test, scores)
        trials.append({'params': params, 'metrics': metrics})
        print(f"trial {index}: PR-AUC {metrics['average_precision_pr_auc']:.6f} "
              f"recall@5%FPR {metrics['official_operating_point']['recall']:.4f} "
              f"recall@100 {metrics['recall_at_100_fp_10k']:.4f}", flush=True)

    best = max(trials, key=lambda t: t['metrics']['average_precision_pr_auc'])
    print('best official trial params', json.dumps(best['params']), flush=True)

    ablations = {}
    for name, columns in variants.items():
        if name == 'official':
            ablations[name] = best['metrics']
            continue
        model = LGBMClassifier(**best['params'], n_jobs=4, verbose=-1, random_state=args.seed)
        model.fit(train[columns], y_train, categorical_feature=CATEGORICAL)
        scores = model.predict_proba(test[columns])[:, 1]
        ablations[name] = evaluate(y_test, scores)
        print(f"{name}: PR-AUC {ablations[name]['average_precision_pr_auc']:.6f} "
              f"recall@100 {ablations[name]['recall_at_100_fp_10k']:.4f}", flush=True)

    report = {
        'mode': 'official-benchmark-reproduction',
        'protocol': 'months 0-5 train, 6-7 test (historical; not independent)',
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'deviations': ['12 random trials instead of 100', 'n_estimators capped at 600',
                       'pandas category dtype instead of LabelEncoder'],
        'class_weight': 'none, as in the published notebook',
        'trials': trials, 'best_trial': best, 'ablations': ablations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote', args.output)


if __name__ == '__main__':
    main()
