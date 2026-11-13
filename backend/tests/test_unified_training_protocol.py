"""Pure protocol tests for the Kaggle-only joint trainer."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_unified import balanced_cap, choose_threshold, ieee, metrics, transformer  # noqa: E402


def test_training_adapter_is_train_fitted_and_handles_unknown_categories():
    train = pd.DataFrame({'amount': [1., 2., 3., 4.], 'kind': ['a', 'a', 'b', 'b']})
    valid = pd.DataFrame({'amount': [2., np.nan], 'kind': ['new', None]})
    pre = transformer(['amount'], ['kind']).fit(train)
    output = pre.transform(valid)
    assert output.shape[0] == 2
    assert np.isfinite(output.toarray() if hasattr(output, 'toarray') else output).all()


def test_capping_preserves_temporal_order_and_validation_metrics_are_complete():
    frame = pd.DataFrame({'time': list(range(20))})
    selected, y = balanced_cap(frame, np.arange(20) % 2, 7)
    assert selected.time.is_monotonic_increasing
    assert len(selected) == len(y) == 7
    truth = np.array([0, 0, 1, 1], dtype=np.int8)
    scores = np.array([.01, .2, .6, .95])
    threshold = choose_threshold(truth, scores)
    report = metrics(truth, scores, threshold)
    assert 0 <= threshold <= 1
    assert set(report) >= {'average_precision_pr_auc', 'roc_auc', 'precision', 'recall', 'f1',
                           'confusion_matrix_tn_fp_fn_tp', 'false_alerts_per_10000_operations'}


def test_ieee_adapter_requires_identity_and_enforces_one_to_one_merge(tmp_path):
    transaction = pd.DataFrame({'TransactionID': list(range(100, 140)), 'TransactionDT': list(range(40)),
        'TransactionAmt': [1.] * 40, 'card1': [1] * 40, 'card2': [1] * 40, 'card3': [1] * 40,
        'card5': [1] * 40, 'addr1': [1] * 40, 'dist1': [1.] * 40, 'ProductCD': ['W'] * 40,
        'card4': ['visa'] * 40, 'card6': ['credit'] * 40, 'P_emaildomain': ['x.test'] * 40,
        'isFraud': [i % 5 == 0 for i in range(40)]})
    transaction.to_csv(tmp_path / 'train_transaction.csv', index=False)
    try:
        ieee(tmp_path, 10)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('identity file is required')
    identity = pd.DataFrame({'TransactionID': list(range(100, 140)), 'DeviceType': ['desktop'] * 40,
                             'DeviceInfo': ['browser'] * 40, 'id_31': ['chrome'] * 40})
    identity.to_csv(tmp_path / 'train_identity.csv', index=False)
    result = ieee(tmp_path, 10)
    assert 'DeviceType' in result[0].columns and result[4][-1] == 'id_31'
    identity.iloc[[0, 0]].to_csv(tmp_path / 'train_identity.csv', index=False)
    try:
        ieee(tmp_path, 10)
    except ValueError as exc:
        assert 'unique' in str(exc)
    else:
        raise AssertionError('duplicate identity IDs must fail')
