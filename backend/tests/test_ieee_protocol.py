"""Fixed chronological holdout and validation-only operating point tests."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_ieee import (calibration_diagnostics, operating_profiles, ranking_metrics, split_chronologically,
                        temporal_drift_diagnostics, threshold_analysis, train)


def test_final_holdout_is_fixed_when_training_size_changes():
    data = pd.DataFrame({'TransactionID': range(400), 'TransactionDT': range(400),
                         'isFraud': [i % 8 == 0 for i in range(400)]})
    (train, validation, final_test), bounds = split_chronologically(data)
    assert len(final_test) == 60
    assert bounds['test_start_dt'] == 340
    assert train.tail(100)['TransactionID'].max() < validation['TransactionID'].min()
    assert validation['TransactionID'].max() < final_test['TransactionID'].min()
    assert final_test['TransactionID'].tolist() == list(range(340, 400))


def test_same_timestamp_never_straddles_boundaries():
    data = pd.DataFrame({'TransactionID': range(400), 'TransactionDT': np.arange(400) // 4,
                         'isFraud': [i % 8 == 0 for i in range(400)]})
    (train, validation, final_test), _ = split_chronologically(data)
    assert train.TransactionDT.max() < validation.TransactionDT.min()
    assert validation.TransactionDT.max() < final_test.TransactionDT.min()


def test_false_alert_budget_uses_exact_validation_counts():
    labels = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    scores = np.array([.9, .85, .8, .7, .6, .4, .3, .2, .1, .05])
    threshold, analysis = threshold_analysis(labels, scores, max_false_alerts_per_10000=0)
    assert threshold == .9
    assert analysis['strategy'] == 'false_alert_budget'
    assert any(row['false_alerts'] > 0 for row in analysis['curve'])
    f1_threshold, _ = threshold_analysis(labels, scores)
    assert f1_threshold <= .9
    with pytest.raises(ValueError, match='nonnegative'):
        threshold_analysis(labels, scores, max_false_alerts_per_10000=-1)


def test_validation_profiles_and_ranking_keep_constraints_explicit():
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=np.int8)
    scores = np.array([.80, .95, .70, .40, .20, .10])
    profiles = operating_profiles(labels, scores, .70, monitoring_min_recall=.8,
                                  high_confidence_min_precision=.99)
    ranking = ranking_metrics(labels, scores, ks=(2, 10))
    assert profiles['analyst_review']['metrics']['cases'] == 6
    assert profiles['monitoring']['constraint_met'] is True
    assert profiles['monitoring']['metrics']['recall'] >= .8
    assert profiles['high_confidence']['available'] is False
    assert ranking['2'] == {'alerts': 2, 'precision_at_k': .5, 'recall_at_k': .333333,
                            'true_positives_at_k': 1}
    assert ranking['10']['alerts'] == 6


def test_calibration_diagnostics_are_bounded_and_not_probability_claims():
    report = calibration_diagnostics(np.array([0, 0, 1, 1]), np.array([.1, .4, .6, .9]), bins=2)
    assert report['brier_score_uncalibrated'] == .085
    assert sum(item['cases'] for item in report['bins']) == 4
    assert 'not calibrated' in report['warning']
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        calibration_diagnostics(np.array([0, 1]), np.array([-.1, 1.1]))


def test_temporal_drift_uses_train_bins_and_reports_unseen_categories():
    train = pd.DataFrame({'amount': [1., 2., 3., np.nan], 'kind': ['a', 'a', 'b', None]})
    validation = pd.DataFrame({'amount': [100., 101., np.nan], 'kind': ['a', 'new', None]})
    report = temporal_drift_diagnostics(train, validation, ['amount'], ['kind'], bins=3)
    assert report['numeric'][0]['psi_train_binned'] is not None
    assert report['categorical'][0]['validation_unseen_rate'] == pytest.approx(1 / 3)
    assert 'does not prove' in report['warning']


def test_existing_report_is_not_overwritten(tmp_path):
    output = tmp_path / 'ieee_final.joblib'
    report = output.with_suffix('.metrics.json')
    report.write_text('keep this result', encoding='utf-8')
    with pytest.raises(FileExistsError, match='Output already exists'):
        train(tmp_path, output, n_estimators=10)
    assert report.read_text(encoding='utf-8') == 'keep this result'
