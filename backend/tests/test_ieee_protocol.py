"""Fixed chronological holdout and validation-only operating point tests."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_ieee import split_chronologically, threshold_analysis, train


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


def test_existing_report_is_not_overwritten(tmp_path):
    output = tmp_path / 'ieee_final.joblib'
    report = output.with_suffix('.metrics.json')
    report.write_text('keep this result', encoding='utf-8')
    with pytest.raises(FileExistsError, match='Output already exists'):
        train(tmp_path, output, n_estimators=10)
    assert report.read_text(encoding='utf-8') == 'keep this result'
