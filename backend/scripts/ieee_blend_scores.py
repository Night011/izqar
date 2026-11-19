"""Blend IEEE-CIS research arms that were scored on identical folds.

`train_ieee_v3.py --save-scores DIR` writes fold{i}_scores.npy / fold{i}_labels.npy.
This script rank-averages several such directories and reports, per fold, PR-AUC
and recall at a fixed false-alert budget, both threshold-free.

Nothing here fits a threshold or uses labels to build the blend: ranks come from
the scores alone, and the budget recall is a post-hoc read of the ranking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def rank_average(score_list):
    """Average within-batch ranks so members need no shared probability scale."""
    ranked = []
    for scores in score_list:
        scores = np.asarray(scores, dtype=np.float64)
        order = np.argsort(np.argsort(scores, kind='stable'), kind='stable')
        ranked.append(order / max(len(order) - 1, 1))
    return np.mean(ranked, axis=0)


def recall_at_budget(y_true, scores, fp_per_10k=100.0):
    """Largest recall among cutoffs whose false-alert rate stays inside the budget."""
    y_true = np.asarray(y_true).astype(int)
    negatives = int((y_true == 0).sum())
    positives = max(int(y_true.sum()), 1)
    allowed = int(np.floor(fp_per_10k * negatives / 10000))
    order = np.argsort(-np.asarray(scores), kind='stable')
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    feasible = fp <= allowed
    if not feasible.any():
        return 0.0, 0.0
    best = int(np.max(tp[feasible]))
    cutoff_index = int(np.max(np.nonzero(feasible)[0]))
    achieved_fp = fp[cutoff_index] * 10000 / max(negatives, 1)
    return round(best / positives, 6), round(float(achieved_fp), 2)


def blend_report(directories, fp_per_10k=100.0, folds=None):
    """PR-AUC and budget recall for the blend of one scores directory per arm."""
    fold_ids = folds if folds is not None else sorted(
        int(path.stem.replace('fold', '').replace('_scores', ''))
        for path in Path(directories[0]).glob('fold*_scores.npy'))
    per_fold = []
    for fold in fold_ids:
        member_scores = [np.load(Path(directory) / f'fold{fold}_scores.npy')
                         for directory in directories]
        y_true = np.load(Path(directories[0]) / f'fold{fold}_labels.npy')
        for other in directories[1:]:
            assert np.array_equal(y_true, np.load(Path(other) / f'fold{fold}_labels.npy')), \
                f'fold {fold}: directories disagree on labels'
        blended = member_scores[0] if len(member_scores) == 1 else rank_average(member_scores)
        recall, achieved_fp = recall_at_budget(y_true, blended, fp_per_10k)
        per_fold.append({'fold': fold,
                         'pr_auc': round(float(average_precision_score(y_true, blended)), 6),
                         'recall_at_budget': recall,
                         'false_alerts_per_10000': achieved_fp})
    mean = {key: round(float(np.mean([row[key] for row in per_fold])), 6)
            for key in ('pr_auc', 'recall_at_budget')}
    return {'arms': [str(d) for d in directories], 'fp_per_10k': fp_per_10k,
            'folds': per_fold, 'mean_metrics': mean}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--fp-per-10k', type=float, default=100.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = blend_report(args.scores_dir, args.fp_per_10k)
    for row in report['folds']:
        print(f"fold {row['fold']}: PR-AUC {row['pr_auc']:.6f} "
              f"recall@{args.fp_per_10k:g} {row['recall_at_budget']:.4f} "
              f"FP/10k {row['false_alerts_per_10000']:.1f}")
    print('mean', report['mean_metrics'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote', args.output)


if __name__ == '__main__':
    main()
