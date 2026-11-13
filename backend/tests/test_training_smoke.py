"""Exercise real training code on tiny generated data; scores are test-only."""
import json
from pathlib import Path
import sys

import pytest

pd = pytest.importorskip('pandas')
pytest.importorskip('sklearn')
pytest.importorskip('xgboost')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_ieee import train as train_ieee  # noqa: E402
from train_paysim import train as train_paysim  # noqa: E402


def test_both_trainers_create_separate_artifacts(tmp_path):
    rows = 300
    paysim = pd.DataFrame({
        'step': range(rows), 'type': ['TRANSFER' if i % 2 else 'PAYMENT' for i in range(rows)],
        'amount': [float(20 + i % 15) for i in range(rows)],
        'oldbalanceOrg': [float(100 + i % 20) for i in range(rows)],
        'oldbalanceDest': [float(i % 20) for i in range(rows)],
        'nameOrig': [f'C{i}' for i in range(rows)], 'nameDest': [f'M{i}' for i in range(rows)],
        'newbalanceOrig': [0.0] * rows, 'newbalanceDest': [0.0] * rows,
        'isFlaggedFraud': [0] * rows, 'isFraud': [int(i % 10 == 0) for i in range(rows)],
    })
    paysim_path = tmp_path / 'paysim.csv'
    paysim.to_csv(paysim_path, index=False)
    pay_model = tmp_path / 'paysim.joblib'
    train_paysim(paysim_path, pay_model, n_estimators=8, max_rows=None)
    pay_report = json.loads(pay_model.with_suffix('.metrics.json').read_text(encoding='utf-8'))
    assert pay_report['model_kind'] == 'paysim'
    assert pay_report['test_metrics']['cases'] == 45
    assert 'isFraud' not in pay_report['features']

    ieee_dir = tmp_path / 'ieee'
    ieee_dir.mkdir()
    pd.DataFrame({
        'TransactionID': range(1000, 1000 + rows), 'TransactionDT': range(rows),
        'TransactionAmt': [float(10 + i % 30) for i in range(rows)],
        'ProductCD': ['W' if i % 2 else 'C' for i in range(rows)],
        'isFraud': [int(i % 10 == 0) for i in range(rows)],
    }).to_csv(ieee_dir / 'train_transaction.csv', index=False)
    pd.DataFrame({'TransactionID': range(1000, 1000 + rows), 'DeviceType': ['desktop'] * rows}).to_csv(ieee_dir / 'train_identity.csv', index=False)
    ieee_model = tmp_path / 'ieee.joblib'
    train_ieee(ieee_dir, ieee_model, n_estimators=8, max_rows=None)
    ieee_report = json.loads(ieee_model.with_suffix('.metrics.json').read_text(encoding='utf-8'))
    assert ieee_report['model_kind'] == 'ieee_cis'
    assert ieee_report['test_metrics']['cases'] == 45
    assert 'TransactionID' not in ieee_report['features']
    assert 'isFraud' not in ieee_report['features']
    assert ieee_report['baseline_convergence']['converged'] is True
    assert ieee_report['baseline_convergence']['solver'] == 'liblinear'
    assert ieee_report['baseline_convergence']['iterations'] < ieee_report['baseline_convergence']['max_iter']
    assert ieee_report['feature_importance']['leakage_audit']
    experiment_model = tmp_path / 'experiments' / 'ieee_100.joblib'
    train_ieee(ieee_dir, experiment_model, n_estimators=8, max_rows=100, mode='experiment')
    experiment_report = json.loads(experiment_model.with_suffix('.metrics.json').read_text(encoding='utf-8'))
    assert not experiment_model.exists()
    assert experiment_report['test_metrics'] is None
    assert experiment_report['baseline_test_metrics'] is None
    assert experiment_report['split_boundaries'] == ieee_report['split_boundaries']
    assert experiment_report['split_counts']['final_test'] == ieee_report['split_counts']['final_test']
    assert experiment_report['split_counts']['train_used'] == 100


def test_ieee_rejects_duplicate_transaction_id(tmp_path):
    pd.DataFrame({'TransactionID': [1, 1], 'TransactionDT': [1, 2], 'TransactionAmt': [10, 20], 'isFraud': [0, 1]}).to_csv(tmp_path / 'train_transaction.csv', index=False)
    pd.DataFrame({'TransactionID': [1], 'DeviceType': ['desktop']}).to_csv(tmp_path / 'train_identity.csv', index=False)
    with pytest.raises(ValueError, match='unique'):
        train_ieee(tmp_path, tmp_path / 'model.joblib', n_estimators=5, max_rows=None)
