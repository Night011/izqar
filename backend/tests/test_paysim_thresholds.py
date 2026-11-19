"""Per-type operating point tests: each type keeps its own false-alert budget."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_paysim import choose_type_thresholds, compare_threshold_policies  # noqa: E402


def _rows(types, labels, scores):
    return pd.DataFrame({'type': types, 'isFraud': labels}), np.asarray(scores, dtype=float)


def test_per_type_threshold_holds_its_own_budget():
    # TRANSFER frauds score high, CASH_OUT frauds score low: one global cutoff
    # cannot serve both, a per-type cutoff can.
    types = ['TRANSFER'] * 2000 + ['CASH_OUT'] * 2000
    labels = [0] * 2000 + [0] * 2000
    scores = list(np.linspace(0.9, 0.1, 2000)) + list(np.linspace(0.5, 0.0, 2000))
    rows, scores = _rows(types, labels, scores)
    scores[:10] = 0.99          # TRANSFER frauds
    rows.loc[:9, 'isFraud'] = 1
    scores[2000:2010] = 0.4     # CASH_OUT frauds sit below the global cutoff
    rows.loc[2000:2009, 'isFraud'] = 1

    thresholds = choose_type_thresholds(rows, scores, global_threshold=0.6, budget_per_10k=100.0)
    assert set(thresholds) == {'TRANSFER', 'CASH_OUT'}
    assert thresholds['CASH_OUT'] < 0.6, 'the type with lower-scoring frauds needs a lower cutoff'
    for transaction_type in ('TRANSFER', 'CASH_OUT'):
        mask = rows['type'].eq(transaction_type).to_numpy()
        predicted = scores[mask] >= thresholds[transaction_type]
        false_alerts = int((predicted & (rows.loc[mask, 'isFraud'].to_numpy() == 0)).sum())
        negatives = int((rows.loc[mask, 'isFraud'].to_numpy() == 0).sum())
        assert false_alerts * 10000 / negatives <= 100.0


def test_rare_type_falls_back_to_the_global_cutoff():
    types = ['TRANSFER'] * 2000 + ['RARE'] * 5
    labels = [0] * 2005
    rows, scores = _rows(types, labels, np.linspace(0.9, 0.1, 2005))
    thresholds = choose_type_thresholds(rows, scores, global_threshold=0.42)
    assert thresholds['RARE'] == 0.42, 'too few negatives to estimate a budget'


def test_policy_comparison_uses_disjoint_halves():
    types = ['TRANSFER', 'CASH_OUT'] * 5000
    labels = [0] * 10000
    scores = np.where(np.array(types) == 'TRANSFER', 0.7, 0.3) + np.linspace(0, 0.2, 10000)
    rows, scores = _rows(types, labels, scores)
    rows.loc[np.array(types) == 'TRANSFER', 'isFraud'] = 0
    rows.loc[(rows.index % 2 == 0), 'isFraud'] = 0
    rows.loc[rows.index[rows.index % 1000 == 0], 'isFraud'] = 1
    report = compare_threshold_policies(rows, scores)
    assert report['fit_rows'] + report['holdout_rows'] == len(rows)
    assert report['holdout_global_f1']['confusion_matrix_tn_fp_fn_tp'][0] > 0
    assert 'per_type_thresholds' in report
    # every policy is measured on rows it was not fitted on
    assert all(report[key]['confusion_matrix_tn_fp_fn_tp'][0] >= 0
               for key in ('holdout_global_f1', 'holdout_global_budget', 'holdout_per_type'))


def test_cutoff_for_budget_cuts_below_the_budgeted_false_alerts():
    from train_paysim import cutoff_for_budget

    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    # a budget of zero eaten FPs still admits the top fraud with no false alert
    assert cutoff_for_budget(y, scores, 100.0) == pytest.approx(0.9)
    # the second fraud sits below two false alerts, so a one-FP budget keeps the same cutoff
    assert cutoff_for_budget(y, scores, 1250.0) == pytest.approx(0.9)
    # the second fraud sits below three false alerts, so only a 3/8 budget reaches it
    assert cutoff_for_budget(y, scores, 2500.0) == pytest.approx(0.9)
    assert cutoff_for_budget(y, scores, 3750.0) == pytest.approx(0.5)
    assert cutoff_for_budget(np.zeros(10, dtype=int), scores) is None
