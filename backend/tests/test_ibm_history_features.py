"""Account history features must be causal and must never expose account IDs."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ibm_history_features import CausalAccountHistory, CausalHistoryStream, HISTORY_FEATURES, build_history_features  # noqa: E402


def test_history_features_use_only_prior_timestamps_and_hide_accounts():
    rows = pd.DataFrame({
        'Timestamp': ['2025-01-01 00:00:00', '2025-01-01 00:00:00', '2025-01-01 00:05:00'],
        'Account': ['A', 'C', 'A'], 'Account.1': ['B', 'B', 'D'], 'Amount Paid': [10, 20, 7],
        'Is Laundering': [1, 0, 1],
    })
    features, history = build_history_features(rows)
    assert list(features.columns) == HISTORY_FEATURES
    # Row 2 has the same time as row 1, so it cannot see row 1's incoming B event.
    assert features.iloc[1].to_dict() == {
        'sender_prior_outgoing_count': 0.0, 'recipient_prior_incoming_count': 0.0,
        'sender_prior_outgoing_amount': 0.0, 'recipient_prior_incoming_amount': 0.0,
        'seconds_since_sender_outgoing': -1.0, 'seconds_since_recipient_incoming': -1.0,
    }
    assert features.iloc[2]['sender_prior_outgoing_count'] == 1.0
    assert features.iloc[2]['sender_prior_outgoing_amount'] == 10.0
    assert features.iloc[2]['seconds_since_sender_outgoing'] == 300.0
    assert history.tracked_accounts == 4
    assert not {'Account', 'Account.1', 'Is Laundering'} & set(features.columns)


def test_history_rejects_nonchronological_and_has_bounded_cold_start():
    rows = pd.DataFrame({
        'Timestamp': ['2025-01-01 00:01:00', '2025-01-01 00:00:00'],
        'Account': ['A', 'C'], 'Account.1': ['B', 'D'], 'Amount Paid': [1, 2],
    })
    with pytest.raises(ValueError, match='chronological'):
        build_history_features(rows)
    history = CausalAccountHistory(max_accounts=1)
    current = pd.DataFrame({'Timestamp': ['2025-01-01 00:00:00'],
                            'Account': ['A'], 'Account.1': ['B'], 'Amount Paid': [1]})
    _, state = build_history_features(current, history)
    assert state.tracked_accounts == 1


def test_history_stream_holds_shared_timestamp_across_chunks():
    stream = CausalHistoryStream()
    first = pd.DataFrame({'Timestamp': ['2025-01-01 00:00:00', '2025-01-01 00:01:00'],
                          'Account': ['A', 'C'], 'Account.1': ['B', 'B'], 'Amount Paid': [10, 20]})
    second = pd.DataFrame({'Timestamp': ['2025-01-01 00:01:00', '2025-01-01 00:02:00'],
                           'Account': ['D', 'A'], 'Account.1': ['B', 'E'], 'Amount Paid': [30, 5]})
    ready_one, features_one = stream.feed(first)
    ready_two, features_two = stream.feed(second)
    ready_three, features_three = stream.finish()
    assert ready_one.Timestamp.tolist() == ['2025-01-01 00:00:00']
    # Both 00:01 rows are returned together and do not see one another.
    assert ready_two.Timestamp.tolist() == ['2025-01-01 00:01:00', '2025-01-01 00:01:00']
    assert features_two.iloc[0]['recipient_prior_incoming_count'] == 1.0
    assert features_two.iloc[1]['recipient_prior_incoming_count'] == 1.0
    assert ready_three.Timestamp.tolist() == ['2025-01-01 00:02:00']
    assert features_three.iloc[0]['sender_prior_outgoing_count'] == 1.0
