"""Regression tests for the corrected BAF evaluation protocol.

The protocol's whole point is that the external fold never influences the
model, the round count or the operating threshold. These tests fail if that
boundary is ever redrawn.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip('pandas')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from baf_protocol import (budget_alert_rate, calibrated_folds, evaluate_external,  # noqa: E402
                          evaluate_external_at_capacity, infold_folds, quantile_cutoff,
                          rows_fingerprint, select_operating_point)
from train_baf import CATEGORICAL, NUMERIC, train  # noqa: E402


def test_calibrated_folds_are_strictly_ordered_and_disjoint():
    folds = calibrated_folds([0, 1, 2, 3, 4, 5])
    assert [fold['external_month'] for fold in folds] == [3, 4, 5]
    for fold in folds:
        train_months = set(fold['train_months'])
        assert max(train_months) < fold['calibration_month'] < fold['external_month']
        assert fold['calibration_month'] not in train_months
        assert fold['external_month'] not in train_months


def test_infold_folds_keep_the_legacy_shape():
    assert infold_folds([0, 1, 2, 3])[0] == {'train_months': [0, 1], 'external_month': 2}
    assert 'calibration_month' not in infold_folds([0, 1, 2, 3])[0]


def test_protocol_rejects_impossible_month_layouts():
    with pytest.raises(ValueError):
        calibrated_folds([0, 1, 2, 3], min_train_months=1)
    with pytest.raises(ValueError):
        calibrated_folds([0, 1])


def test_threshold_comes_from_calibration_and_is_frozen_on_external():
    rng = np.random.default_rng(0)
    y_calibration = (rng.random(2000) < 0.05).astype(int)
    scores_calibration = np.clip(0.8 * y_calibration + rng.random(2000) * 0.6, 0, 1)
    threshold, selection = select_operating_point(y_calibration, scores_calibration, 50)
    assert selection['selection_data'] == 'calibration month only'
    assert selection['achieved_false_alerts_per_10000_on_calibration'] <= 50

    # External period is deliberately shifted: it may exceed the budget and that
    # must be reported, not repaired by re-fitting the threshold.
    y_external = (rng.random(2000) < 0.15).astype(int)
    scores_external = np.clip(0.8 * y_external + rng.random(2000) * 0.6, 0, 1)
    external = evaluate_external(y_external, scores_external, threshold)
    predicted = scores_external >= threshold
    expected_fp = int(np.sum(predicted & (y_external == 0)))
    assert external['threshold'] == round(float(threshold), 6)
    assert external['threshold_source'] == 'frozen from calibration month'
    assert external['confusion_matrix_tn_fp_fn_tp'][1] == expected_fp
    assert set(external['post_hoc_budget_diagnostics']) == {'50', '100', '200', '500'}


def test_capacity_policy_replays_alert_rate_without_labels():
    rng = np.random.default_rng(1)
    y_calibration = (rng.random(4000) < 0.05).astype(int)
    scores_calibration = np.clip(0.8 * y_calibration + rng.random(4000) * 0.6, 0, 1)
    rate = budget_alert_rate(y_calibration, scores_calibration, 50)
    assert 0 < rate < 1

    # Same score scale, more benign traffic: the policy flags the same share,
    # so the false-alert count is what changes, not the cutoff rule.
    y_external = (rng.random(4000) < 0.05).astype(int)
    scores_external = np.clip(0.8 * y_external + rng.random(4000) * 0.6, 0, 1)
    external = evaluate_external_at_capacity(y_external, scores_external, rate)
    assert external['frozen_alert_rate'] == pytest.approx(rate, abs=1e-6)
    assert external['achieved_alert_rate'] == pytest.approx(rate, abs=0.002)
    assert external['threshold_source'].startswith('frozen alert rate')


def test_quantile_cutoff_uses_scores_only():
    scores = np.arange(1000, dtype=float)
    cutoff = quantile_cutoff(scores, 0.1)
    assert int(np.sum(scores >= cutoff)) == 100
    with pytest.raises(ValueError):
        quantile_cutoff(scores, 0.0)


def test_rows_fingerprint_tracks_the_exact_rows():
    frame = pd.DataFrame({'month': [1, 1, 2], 'fraud_bool': [0, 1, 0]},
                         index=[10, 11, 12])
    first = rows_fingerprint(frame)
    assert first == {'rows': 3, 'fraud_cases': 1, 'fraud_rate': pytest.approx(1 / 3),
                     'row_index_sha256': first['row_index_sha256']}
    assert rows_fingerprint(frame.iloc[:2])['row_index_sha256'] != first['row_index_sha256']


def make_base(path: Path, rows_per_month: int = 30, months: int = 8):
    size = rows_per_month * months
    data = {'month': [index // rows_per_month for index in range(size)],
            'fraud_bool': [int(index % 5 == 0) for index in range(size)],
            'device_fraud_count': [int(index % 5 == 0) for index in range(size)],
            'days_since_request': [int(index % 5 == 0) for index in range(size)]}
    for name in NUMERIC:
        data[name] = [float(index % 15 + 1) for index in range(size)]
    for name in CATEGORICAL:
        data[name] = ['a' if index % 2 else 'b' for index in range(size)]
    pd.DataFrame(data).to_csv(path, index=False)


def test_research_run_uses_three_disjoint_periods(tmp_path):
    path = tmp_path / 'Base.csv'
    make_base(path)
    output = tmp_path / 'research.metrics.json'
    report = train(path, output, n_estimators=8, mode='research', protocol='calibrated',
                   max_false_alerts_per_10000=100, early_stopping_rounds=2)

    assert report['protocol'] == 'calibrated'
    assert [fold['external_month'] for fold in report['folds']] == [3, 4, 5]
    used_months = set()
    for fold in report['folds']:
        assert fold['selection']['selection_data'] == 'calibration month only'
        assert fold['external_metrics']['threshold_source'] == 'frozen from calibration month'
        assert fold['selected_threshold'] == fold['external_metrics']['threshold']
        assert fold['rounds_used'] == fold['best_iteration'] + 1
        assert fold['checksums']['external']['row_index_sha256']
        used_months |= set(fold['train_months'])
        used_months |= {fold['calibration_month'], fold['external_month']}
    assert used_months == {0, 1, 2, 3, 4, 5}
    saved = json.loads(output.read_text(encoding='utf-8'))
    assert saved['folds'] == report['folds']


def test_legacy_protocol_is_labelled_as_optimistic(tmp_path):
    path = tmp_path / 'Base.csv'
    make_base(path)
    report = train(path, tmp_path / 'legacy.metrics.json', n_estimators=8, mode='research',
                   protocol='infold', max_false_alerts_per_10000=100, early_stopping_rounds=2)
    assert report['protocol'] == 'infold'
    assert report['folds'][0]['calibration_month'] is None
    assert report['folds'][0]['selection']['selection_data'] == 'external fold (legacy, optimistic)'
    assert 'not valid out-of-fold' in report['selection_rule']


def test_research_rejects_unknown_protocol(tmp_path):
    with pytest.raises(ValueError, match='protocol'):
        train(tmp_path / 'not-read.csv', tmp_path / 'out.json', mode='research', protocol='oracle')
