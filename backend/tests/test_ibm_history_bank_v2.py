"""Bank-scoped history is an independent validation-only experiment.

Existing account_only_v1 output remains unchanged, preserving old model
semantics and the previously published benchmarks.
"""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ibm_history_features import HISTORY_FEATURES, CausalHistoryStream, build_history_features  # noqa: E402


def bank_rows():
    return pd.DataFrame({
        'Timestamp': ['2025-01-01 00:00:00', '2025-01-01 00:01:00',
                      '2025-01-01 00:01:00', '2025-01-01 00:02:00'],
        'From Bank': ['001', '002', '001', '001'],
        'Account': ['A', 'A', 'A', 'A'],
        'To Bank': ['005', '005', '006', '007'],
        'Account.1': ['B', 'C', 'D', 'E'],
        'Amount Paid': [10, 20, 30, 40],
    })


def test_bank_key_avoids_cross_bank_account_collision_and_preserves_legacy():
    rows = bank_rows()
    legacy, _ = build_history_features(rows)
    banked, state = build_history_features(rows, identity_mode='bank_account_v2')
    # Prior transfers from bank 001 account A must not pollute bank 002 A.
    assert legacy.iloc[1]['sender_prior_outgoing_count'] == 1
    assert banked.iloc[1]['sender_prior_outgoing_count'] == 0
    assert banked.iloc[1]['seconds_since_sender_outgoing'] == -1
    assert banked.iloc[2]['sender_prior_outgoing_count'] == 1
    assert banked.iloc[2]['sender_prior_outgoing_amount'] == 10
    assert banked.iloc[3]['sender_prior_outgoing_count'] == 2
    assert banked.iloc[3]['sender_prior_outgoing_amount'] == 40
    assert state.tracked_accounts >= 2
    assert list(banked.columns) == HISTORY_FEATURES
    assert not set(rows.columns) & set(banked.columns)


def test_stream_preserves_timestamp_batch_semantics_in_v2():
    rows = bank_rows()
    stream = CausalHistoryStream(identity_mode='bank_account_v2')
    a, features_a = stream.feed(rows.iloc[:2])
    b, features_b = stream.feed(rows.iloc[2:3])
    c, features_c = stream.feed(rows.iloc[3:])
    d, features_d = stream.finish()
    assert len(a) == 1 and len(b) == 0 and len(c) == 2 and len(d) == 1
    assert features_c.iloc[0]['sender_prior_outgoing_count'] == 0  # 002 A
    assert features_c.iloc[1]['sender_prior_outgoing_count'] == 1  # 001 A
    assert features_d.iloc[0]['sender_prior_outgoing_count'] == 2
    assert features_a.iloc[0]['sender_prior_outgoing_count'] == 0


def test_banked_history_rejects_missing_bank_and_unknown_mode():
    rows = bank_rows()
    with pytest.raises(ValueError, match='missing historical'):
        build_history_features(rows.drop(columns='From Bank'), identity_mode='bank_account_v2')
    with pytest.raises(ValueError, match='Unknown'):
        build_history_features(rows, identity_mode='random')
    rows.loc[0, 'From Bank'] = None
    with pytest.raises(ValueError, match='bank identifiers'):
        build_history_features(rows, identity_mode='bank_account_v2')
    with pytest.raises(ValueError, match='Unknown'):
        CausalHistoryStream(identity_mode='random')


def test_bank_causal_candidate_cannot_run_final_without_reading_csv(tmp_path):
    from train_ibm_aml import train
    with pytest.raises(ValueError, match='validation-only candidate'):
        train(tmp_path / 'missing.csv', tmp_path / 'forbidden.joblib', mode='final',
              feature_set='causal_history_bank_v2')


def test_shadow_bank_v2_is_distinguishable_from_old_v1(tmp_path):
    from test_ibm_shadow_bundle import fitted_candidate
    from ibm_shadow_bundle import export_shadow_bundle, verify_shadow_bundle
    pipeline, report = fitted_candidate()
    report['experiment']['hyperparameters']['feature_set'] = 'causal_history_bank_v2'
    report['model_version'] = 'ibm-aml-causal-bank-shadow-v2'
    archive = tmp_path / 'bank_v2.zip'
    details = export_shadow_bundle(pipeline, report, archive)
    verified = verify_shadow_bundle(archive, details['sha256'])
    assert verified['stem'] == 'bank_v2'
    assert verified['validation_metrics']['average_precision_pr_auc'] == 0.12


def test_future_labels_and_future_amounts_cannot_change_earlier_banked_history():
    original = bank_rows()
    original['Is Laundering'] = [0, 1, 0, 1]
    before, _ = build_history_features(original, identity_mode='bank_account_v2')
    modified = original.copy()
    modified.loc[3, 'Is Laundering'] = 0
    modified.loc[3, 'Amount Paid'] = 999999
    after, _ = build_history_features(modified, identity_mode='bank_account_v2')
    pd.testing.assert_frame_equal(before.iloc[:3], after.iloc[:3])


def test_account_keys_are_tuples_not_ambiguous_joined_strings():
    rows = pd.DataFrame({
        'Timestamp': ['2025-01-01 00:00:00', '2025-01-01 00:01:00'],
        'From Bank': ['B|X', 'B'], 'Account': ['C', 'X|C'],
        'To Bank': ['R', 'R'], 'Account.1': ['1', '2'], 'Amount Paid': [10, 5],
    })
    features, _ = build_history_features(rows, identity_mode='bank_account_v2')
    assert features.iloc[1]['sender_prior_outgoing_count'] == 0


def test_stream_rejects_reordered_chunks_or_updates_after_finish():
    rows = bank_rows()
    stream = CausalHistoryStream(identity_mode='bank_account_v2')
    stream.feed(rows.iloc[0:1])
    stream.feed(rows.iloc[1:3])
    with pytest.raises(ValueError, match='global chronological'):
        stream.feed(rows.iloc[0:1])
    stream.finish()
    with pytest.raises(ValueError, match='after finish'):
        stream.feed(rows.iloc[3:4])
