"""Evaluate rank-average ensembles from the saved out-of-fold scores.

research_baf_algorithms.py --scores-dir writes, per arm and external month, the
calibration-month and external-month scores plus both label vectors. Every
external score comes from a model that never saw that month, so combining them
uses no in-sample prediction.

Combination is a weighted average of within-batch ranks: it needs no labels, and
the same recipe applies at serving time as long as the batch of scores exists.
The operating point is the calibration alert rate (fixed-capacity policy), so
nothing here is fitted on external labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baf_protocol import (budget_alert_rate, evaluate_external_at_capacity,  # noqa: E402
                          mean_external_metrics)


def rank_normalise(scores):
    order = np.argsort(np.argsort(np.asarray(scores, dtype=np.float64), kind='stable'), kind='stable')
    return order / max(len(order) - 1, 1)


def combine(score_list, weights):
    matrix = np.vstack([rank_normalise(scores) for scores in score_list])
    return np.average(matrix, axis=0, weights=weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores-dir', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', required=True, help='arm prefixes, e.g. lightgbm_v1v')
    parser.add_argument('--months', nargs='+', type=int, default=[3, 4, 5])
    parser.add_argument('--weights', nargs='+', type=float)
    parser.add_argument('--max-false-alerts-per-10000', type=float, default=100.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    weights = args.weights or [1.0] * len(args.arms)
    if len(weights) != len(args.arms):
        raise ValueError('weights must match arms')

    folds = []
    for month in args.months:
        stems = [f'{arm}_m{month}' for arm in args.arms]
        external_scores = [np.load(args.scores_dir / f'{stem}_external.npy') for stem in stems]
        calibration_scores = [np.load(args.scores_dir / f'{stem}_calibration.npy') for stem in stems]
        y_external = np.load(args.scores_dir / f'{stems[0]}_labels.npy')
        y_calibration = np.load(args.scores_dir / f'{stems[0]}_cal_labels.npy')

        rate = budget_alert_rate(y_calibration, combine(calibration_scores, weights),
                                 args.max_false_alerts_per_10000)
        metrics = evaluate_external_at_capacity(y_external, combine(external_scores, weights), rate)
        folds.append({'external_month': month, 'capacity_alert_rate': round(float(rate), 6),
                      'external_metrics_capacity': metrics})
        print(f"month {month}: PR-AUC {metrics['average_precision_pr_auc']:.6f} "
              f"recall {metrics['recall']:.4f} FP/10k "
              f"{metrics['false_alerts_per_10000_operations']:.1f}", flush=True)

    report = {'mode': 'rank-average-ensemble',
              'trained_at_utc': datetime.now(timezone.utc).isoformat(),
              'arms': args.arms, 'weights': weights,
              'operating_point': 'calibration alert rate replayed on external ranks',
              'folds': folds, 'mean_metrics': mean_external_metrics(folds, 'external_metrics_capacity')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('mean PR-AUC', report['mean_metrics']['average_precision_pr_auc'],
          'recall', report['mean_metrics']['recall'])
    print('wrote', args.output)


if __name__ == '__main__':
    main()
