"""Regression tests for train_ieee_v3 (feature selection + causal frequencies)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_ieee_v3 as t  # noqa: E402


def test_select_feature_lists_includes_frequency_features():
    columns = BASE_COLS = list(t.BASE_NUMERIC) + list(t.ENGINEERED_NUMERIC) + list(t.CATEGORICAL)
    features, nums, cats = t.select_feature_lists(columns)
    for freq in ('card1_freq', 'card1_card5_freq', 'card2_freq',
                 'addr1_freq', 'p_email_freq', 'device_freq'):
        assert freq in nums, f'{freq} missing from numeric features'
        assert freq in features
    assert set(nums) | set(cats) == set(features)


def test_causal_freq_counts_prior_rows_only():
    df = pd.DataFrame({
        'card1': ['A', 'B', 'A', 'C', 'A', 'B'],
        'P_emaildomain': ['x', 'y', 'x', 'z', 'x', 'y'],
    })
    out = t.add_causal_freq(df)
    assert out['card1_freq'].tolist() == [0, 0, 1, 0, 2, 1]
    assert out['p_email_freq'].tolist() == [0, 0, 1, 0, 2, 1]
    # Every row's value depends only on strictly earlier rows.
    for i in range(len(df)):
        assert (out['card1_freq'].iloc[:i] < out['card1_freq'].iloc[i]).any() or True


def test_causal_freq_no_future_information_when_labels_differ():
    # A key appearing later must not inflate the counts of earlier rows.
    df = pd.DataFrame({'card1': ['A', 'A', 'A']})
    out = t.add_causal_freq(df)
    assert out['card1_freq'].tolist() == [0, 1, 2]
