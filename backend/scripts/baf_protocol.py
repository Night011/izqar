"""Temporal evaluation protocol shared by BAF research scripts.

The earlier research protocol selected the operating threshold (and the number
of boosting rounds) on the very fold it then reported. That inflates recall,
F1 and the achieved false-alert rate, because the threshold is fitted to the
labels of the evaluation period.

This module implements the corrected protocol. Each external fold has three
strictly ordered, disjoint roles:

    train months          calibration month        external fold
    months[:i-1]          months[i-1]              months[i]

The model is fitted only on the train months. The number of boosting rounds
and the operating threshold are chosen on the calibration month. The external
month is scored once, with both choices frozen, and is never passed to any
selection routine.

Months are always taken from the caller-supplied train region, so the
penultimate and final BAF months (the historical validation/test months) can
never enter model, round, or threshold selection.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

BUDGET_DIAGNOSTICS = (50.0, 75.0, 100.0)


def infold_folds(months: Iterable[int], min_train_months: int = 2) -> list[dict]:
    """Legacy protocol kept for reproducing old (optimistic) numbers.

    Threshold and early stopping are selected on the external fold itself.
    """
    months = list(months)
    _validate(months, min_train_months)
    return [{'train_months': months[:index], 'external_month': months[index]}
            for index in range(min_train_months, len(months))]


def calibrated_folds(months: Iterable[int], min_train_months: int = 2) -> list[dict]:
    """Corrected protocol: train < calibration < external, all disjoint."""
    months = list(months)
    _validate(months, min_train_months)
    folds = []
    for index in range(min_train_months + 1, len(months)):
        folds.append({'train_months': months[:index - 1],
                      'calibration_month': months[index - 1],
                      'external_month': months[index]})
    return folds


def _validate(months: list[int], min_train_months: int) -> None:
    if len(set(months)) != len(months):
        raise ValueError('fold months must be unique')
    if min_train_months < 2:
        raise ValueError('at least two train months are required')
    if len(months) < min_train_months + 1:
        raise ValueError('not enough months for the requested protocol')


def rows_fingerprint(frame) -> dict:
    """Checksum of the exact rows used, for reproducibility auditing."""
    digest = hashlib.sha256()
    digest.update(np.asarray(frame.index.to_numpy(), dtype='<i8').tobytes())
    if 'month' in frame.columns:
        digest.update(np.asarray(frame['month'].to_numpy(), dtype='<i1').tobytes())
    return {'rows': int(len(frame)),
            'fraud_cases': int(frame['fraud_bool'].sum()),
            'fraud_rate': round(float(frame['fraud_bool'].mean()), 6),
            'row_index_sha256': digest.hexdigest()}


def select_operating_point(y_calibration, calibration_scores,
                           max_false_alerts_per_10000: float | None):
    """Choose threshold and report calibration-only operating statistics.

    Only calibration labels are accepted here on purpose: the external fold
    must not influence the operating point.
    """
    from train_ieee import threshold_analysis

    threshold, report = threshold_analysis(
        y_calibration, calibration_scores, max_false_alerts_per_10000)
    y = np.asarray(y_calibration, dtype=np.int8)
    scores = np.asarray(calibration_scores, dtype=np.float64)
    predicted = scores >= threshold
    false_alerts = int(np.sum(predicted & (y == 0)))
    return float(threshold), {
        'selection_data': 'calibration month only',
        'max_false_alerts_per_10000': max_false_alerts_per_10000,
        'achieved_false_alerts_per_10000_on_calibration':
            round(false_alerts * 10000 / max(len(y), 1), 3),
        'calibration_cases': int(len(y)),
        'calibration_fraud_cases': int(y.sum()),
        'curve': report.get('curve', []),
        'max_f1_threshold': report.get('max_f1_threshold'),
        'strategy': report.get('strategy'),
    }


def evaluate_external(y_external, external_scores, threshold,
                      budgets=BUDGET_DIAGNOSTICS) -> dict:
    """Frozen-threshold evaluation plus a clearly labelled post-hoc budget view."""
    from train_ieee import metrics, ranking_metrics, threshold_analysis

    result = metrics(y_external, external_scores, threshold)
    result.update(ranking_metrics(y_external, external_scores))
    result['threshold'] = round(float(threshold), 6)
    result['threshold_source'] = 'frozen from calibration month'
    budget_view = {}
    for budget in budgets:
        if budget is None:
            continue
        cutoff, _ = threshold_analysis(y_external, external_scores, budget)
        point = metrics(y_external, external_scores, cutoff)
        budget_view[str(int(budget))] = {
            'threshold': round(float(cutoff), 6),
            'recall': point['recall'], 'precision': point['precision'],
            'f1': point['f1'],
            'achieved_false_alerts_per_10000': point['false_alerts_per_10000_operations'],
            'note': 'post-hoc diagnostic only; not used to select the model or threshold',
        }
    result['post_hoc_budget_diagnostics'] = budget_view
    return result


def budget_alert_rate(y_calibration, calibration_scores,
                      max_false_alerts_per_10000: float | None) -> float:
    """Share of a batch flagged by the calibration budget threshold.

    The rate is a property of the score distribution, so it can be replayed on
    a later batch without labels: production keeps a rolling reference window
    and flags the same top share of today's applications.
    """
    threshold, _ = select_operating_point(y_calibration, calibration_scores,
                                          max_false_alerts_per_10000)
    scores = np.asarray(calibration_scores, dtype=np.float64)
    return float(np.mean(scores >= threshold))


def quantile_cutoff(scores, alert_rate: float) -> float:
    """Score cutoff flagging the top ``alert_rate`` share of a batch. No labels."""
    if not 0.0 < alert_rate <= 1.0:
        raise ValueError('alert_rate must be in (0, 1]')
    scores = np.asarray(scores, dtype=np.float64)
    count = min(max(int(np.ceil(alert_rate * len(scores))), 1), len(scores))
    return float(np.partition(scores, len(scores) - count)[len(scores) - count])


def evaluate_external_at_capacity(y_external, external_scores, alert_rate: float) -> dict:
    """Frozen-capacity operating point: fixed alert volume, threshold from scores only."""
    from train_ieee import metrics, ranking_metrics

    cutoff = quantile_cutoff(external_scores, alert_rate)
    result = metrics(y_external, external_scores, cutoff)
    result.update(ranking_metrics(y_external, external_scores))
    result['threshold'] = round(cutoff, 6)
    result['threshold_source'] = 'frozen alert rate replayed on external scores (no labels)'
    result['frozen_alert_rate'] = round(float(alert_rate), 6)
    result['achieved_alert_rate'] = round(float(np.mean(np.asarray(external_scores) >= cutoff)), 6)
    return result


def mean_external_metrics(fold_reports: list[dict], key: str = 'external_metrics') -> dict:
    """Average numeric metrics across folds (post-hoc views excluded)."""
    if not fold_reports:
        return {}
    reports = [fold[key] for fold in fold_reports if key in fold]
    if not reports:
        return {}
    scalar_keys = [name for name, value in reports[0].items()
                   if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {name: round(float(np.mean([report[name] for report in reports])), 6)
            for name in scalar_keys}
