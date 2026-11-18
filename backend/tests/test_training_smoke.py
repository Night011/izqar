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
from train_paysim import train as train_paysim, validate_run_configuration as validate_paysim_run_configuration  # noqa: E402
from audit_ieee_validation import audit as audit_ieee_validation  # noqa: E402
from audit_paysim_validation import audit as audit_paysim_validation  # noqa: E402
from export_portable_model import export as export_portable_model  # noqa: E402
from install_model_bundle import install as install_model_bundle  # noqa: E402


def test_paysim_engineered_final_requires_explicit_freeze():
    with pytest.raises(ValueError, match='explicit matching frozen'):
        validate_paysim_run_configuration('final', 1.0, 'pre_transaction_v2', None)
    with pytest.raises(ValueError, match='exactly match'):
        validate_paysim_run_configuration('final', 1.0, 'pre_transaction_v2', 'pre_transaction_v1')
    with pytest.raises(ValueError, match='valid only in final mode'):
        validate_paysim_run_configuration('experiment', 1.0, 'pre_transaction_v2', 'pre_transaction_v2')
    validate_paysim_run_configuration('final', 1.0, 'pre_transaction_v2', 'pre_transaction_v2')


def test_ieee_extended_candidate_cannot_consume_inspected_final_holdout(tmp_path):
    with pytest.raises(ValueError, match='validation-only'):
        train_ieee(tmp_path / 'not-read', tmp_path / 'final.joblib', n_estimators=8,
                   mode='final', feature_set='extended_cv')


def test_both_trainers_create_separate_artifacts(tmp_path, client, auth, monkeypatch):
    rows = 300
    paysim = pd.DataFrame({
        'step': range(rows), 'type': [['TRANSFER', 'CASH_OUT', 'PAYMENT'][i % 3] for i in range(rows)],
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
    assert pay_report['baseline_convergence']['converged'] is True
    assert set(pay_report['validation_operating_profiles']) == {'monitoring', 'analyst_review', 'high_confidence'}
    assert {'100', '500', '1000'} <= set(pay_report['validation_ranking_metrics'])
    assert pay_report['run_mode'] == 'final'
    assert pay_report['validation_type_slices']['TRANSFER']['available'] is True
    assert pay_report['validation_type_slices']['CASH_OUT']['available'] is True
    assert pay_report['validation_robustness']['missing_balances_every_tenth_row']['affected_rows'] > 0
    assert pay_report['validation_robustness']['unseen_type_encoding']['cases'] == pay_report['split_counts']['validation']
    assert pay_report['validation_temporal_drift']['numeric'][0]['feature'] in pay_report['numerical_features']
    assert pay_report['feature_importance']['top_30']
    assert all('isFraud' not in row['feature'] and 'newbalance' not in row['feature'] for row in pay_report['feature_importance']['top_30'])
    pay_experiment = tmp_path / 'experiments' / 'paysim_candidate.joblib'
    train_paysim(paysim_path, pay_experiment, n_estimators=8, max_rows=None, mode='experiment')
    pay_experiment_report = json.loads(pay_experiment.with_suffix('.metrics.json').read_text(encoding='utf-8'))
    assert not pay_experiment.exists()
    assert pay_experiment_report['run_mode'] == 'experiment'
    assert pay_experiment_report['test_metrics'] is None
    assert pay_experiment_report['baseline_test_metrics'] is None
    weighted_pay = tmp_path / 'experiments' / 'paysim_weighted.joblib'
    weighted_pay_report = train_paysim(paysim_path, weighted_pay, n_estimators=8, max_rows=None,
                                       mode='experiment', positive_weight_multiplier=1.5)
    assert weighted_pay_report['experiment']['hyperparameters']['positive_weight_multiplier'] == 1.5
    early_stop_pay = tmp_path / 'experiments' / 'paysim_early_stop.joblib'
    early_stop_report = train_paysim(paysim_path, early_stop_pay, n_estimators=8, max_rows=None,
                                     mode='experiment', early_stopping_rounds=2)
    assert early_stop_report['xgboost_training']['early_stopping_rounds'] == 2
    assert early_stop_report['xgboost_training']['selection_data'].startswith('validation only')
    with pytest.raises(ValueError, match='validation-only'):
        train_paysim(paysim_path, tmp_path / 'paysim_weighted_final.joblib', 8, None,
                     positive_weight_multiplier=1.5)
    with pytest.raises(ValueError, match='validation-only'):
        train_paysim(paysim_path, tmp_path / 'paysim_early_stop_final.joblib', 8, None,
                     early_stopping_rounds=2)
    engineered_pay = tmp_path / 'experiments' / 'paysim_engineered.joblib'
    engineered_report = train_paysim(paysim_path, engineered_pay, n_estimators=8, max_rows=None,
                                     mode='experiment', feature_set='pre_transaction_v2')
    assert not engineered_pay.exists()
    assert engineered_report['experiment']['hyperparameters']['feature_set'] == 'pre_transaction_v2'
    assert 'amount_to_origin_balance' in engineered_report['features']
    with pytest.raises(ValueError, match='explicit matching frozen'):
        train_paysim(paysim_path, tmp_path / 'paysim_engineered_final.joblib', 8, None,
                     feature_set='pre_transaction_v2')
    with pytest.raises(FileExistsError, match='Output already exists'):
        train_paysim(paysim_path, pay_model, 8, None)
    pay_audit = audit_paysim_validation(paysim_path, pay_model, pay_model.with_name(pay_model.stem + '.validation_audit.json'))
    assert pay_audit['analysis_source'] == 'frozen_paysim_validation_only'
    assert pay_audit['cases'] == pay_report['split_counts']['validation']
    assert 'test_metrics' not in pay_audit

    ieee_dir = tmp_path / 'ieee'
    ieee_dir.mkdir()
    pd.DataFrame({
        'TransactionID': range(1000, 1000 + rows), 'TransactionDT': range(rows),
        'TransactionAmt': [float(10 + i % 30) for i in range(rows)],
        'C3': [float(i % 5) for i in range(rows)], 'V4': [float(i % 3) for i in range(rows)],
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
    assert set(ieee_report['validation_operating_profiles']) == {'monitoring', 'analyst_review', 'high_confidence'}
    experiment_model = tmp_path / 'experiments' / 'ieee_100.joblib'
    train_ieee(ieee_dir, experiment_model, n_estimators=8, max_rows=100, mode='experiment')
    experiment_report = json.loads(experiment_model.with_suffix('.metrics.json').read_text(encoding='utf-8'))
    assert not experiment_model.exists()
    assert experiment_report['test_metrics'] is None
    assert experiment_report['baseline_test_metrics'] is None
    assert experiment_report['split_boundaries'] == ieee_report['split_boundaries']
    assert experiment_report['split_counts']['final_test'] == ieee_report['split_counts']['final_test']
    assert experiment_report['split_counts']['train_used'] == 100
    weighted_ieee = tmp_path / 'experiments' / 'ieee_weighted.joblib'
    weighted_ieee_report = train_ieee(ieee_dir, weighted_ieee, n_estimators=8, max_rows=100,
                                      mode='experiment', positive_weight_multiplier=.75)
    assert weighted_ieee_report['experiment']['hyperparameters']['positive_weight_multiplier'] == .75
    extended_path = tmp_path / 'ieee_extended.joblib'
    train_ieee(ieee_dir, extended_path, 8, 100, 'experiment', feature_set='extended_cv')
    extended_report = json.loads(extended_path.with_suffix('.metrics.json').read_text(encoding='utf-8'))
    assert {'C3', 'V4'} <= set(extended_report['features'])
    assert extended_report['test_metrics'] is None
    assert not extended_path.exists()
    audit_path = tmp_path / 'ieee_validation_audit.json'
    audit_report = audit_ieee_validation(ieee_dir, ieee_model, audit_path)
    assert audit_report['analysis_source'] == 'frozen_ieee_validation_only'
    assert audit_report['cases'] == ieee_report['split_counts']['validation']
    assert 'test_metrics' not in audit_report
    assert audit_report['precision_at_100'] >= 0
    import app.ml as ml
    with monkeypatch.context() as patch:
        patch.setattr(ml, 'MODEL_PATH', ieee_model)
        patch.setattr(ml, 'PAYSIM_MODEL_PATH', pay_model)
        ml.load_model.cache_clear()
        ml.load_paysim_model.cache_clear()
        status = client.get('/api/v1/model/status', headers=auth)
        assert status.status_code == 200
        assert status.json()['ieee_cis']['model_version'] == 'ieee-cis-xgboost-v2'
        assert status.json()['paysim']['model_version'] == 'paysim-xgboost-v1'
        assert status.json()['ieee_cis']['validation_ranking_metrics']['100']['alerts'] > 0
        assert status.json()['paysim']['validation_operating_profiles']['analyst_review']['metrics']['cases'] > 0
        assert status.json()['paysim']['validation_type_slices']['TRANSFER']['available'] is True
        assert status.json()['paysim']['validation_robustness']['scope'] == 'validation_only_synthetic_preprocessing_stress'
        assert status.json()['paysim']['validation_temporal_drift']['numeric'][0]['feature'] in pay_report['numerical_features']
        pay_features = {'type': 'TRANSFER', 'amount': 25, 'oldbalanceOrg': 110, 'oldbalanceDest': 5}
        pay_response = client.post('/api/v1/paysim/predict', json={'features': pay_features}, headers=auth)
        assert pay_response.status_code == 200, pay_response.text
        assert pay_response.json()['source'] == 'paysim_ml'
        portable_zip = export_portable_model(pay_model, tmp_path / 'paysim_portable')
        installed = install_model_bundle(portable_zip, tmp_path / 'installed')
        assert len(installed) == 4  # model, preprocessing, report, validation audit
        patch.setattr(ml, 'PAYSIM_MODEL_PATH', tmp_path / 'installed' / 'paysim_portable.metrics.json')
        ml.load_paysim_model.cache_clear()
        portable_response = client.post('/api/v1/paysim/predict', json={'features': pay_features}, headers=auth)
        assert portable_response.status_code == 200, portable_response.text
        assert portable_response.json()['risk_score'] == pay_response.json()['risk_score']
        ieee_features = {key: None for key in ieee_report['features']}
        ieee_features['TransactionAmt'] = 25
        ieee_response = client.post('/api/v1/ieee/predict', json={'features': ieee_features}, headers=auth)
        assert ieee_response.status_code == 200, ieee_response.text
        assert ieee_response.json()['source'] == 'ieee_cis_ml'
    ml.load_model.cache_clear()
    ml.load_paysim_model.cache_clear()


def test_ieee_rejects_duplicate_transaction_id(tmp_path):
    pd.DataFrame({'TransactionID': [1, 1], 'TransactionDT': [1, 2], 'TransactionAmt': [10, 20], 'isFraud': [0, 1]}).to_csv(tmp_path / 'train_transaction.csv', index=False)
    pd.DataFrame({'TransactionID': [1], 'DeviceType': ['desktop']}).to_csv(tmp_path / 'train_identity.csv', index=False)
    with pytest.raises(ValueError, match='unique'):
        train_ieee(tmp_path, tmp_path / 'model.joblib', n_estimators=5, max_rows=None)
