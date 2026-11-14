"""Tests for schema boundaries and leakage guards; no external datasets required."""
from pathlib import Path
import sys
import numpy as np
import pytest

pd = pytest.importorskip('pandas', reason='ML dependency not installed')
pytest.importorskip('sklearn', reason='ML dependency not installed')
pytest.importorskip('xgboost', reason='ML dependency not installed')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_paysim import FEATURES, decision_time_features, feature_schema, read_paysim  # noqa: E402


def test_incompatible_artifact_is_reported_without_unpickling(tmp_path, client, auth, monkeypatch):
    import json
    import app.ml as ml
    model = tmp_path / 'paysim_model.joblib'
    model.write_bytes(b'not a pickle')
    model.with_suffix('.metrics.json').write_text(json.dumps({
        'model_kind': 'paysim', 'library_versions': {'scikit_learn': '0.0.0'},
    }), encoding='utf-8')
    with monkeypatch.context() as patch:
        patch.setattr(ml, 'PAYSIM_MODEL_PATH', model)
        # This test replaces joblib.load globally. Keep the separately installed
        # Unified artifact out of the status endpoint so the assertion remains
        # scoped to PaySim's pre-unpickle compatibility gate.
        patch.setattr(ml, 'UNIFIED_MODEL_PATH', tmp_path / 'missing_unified.metrics.json')
        patch.setattr(ml.joblib, 'load', lambda *_: pytest.fail('must reject before unpickling'))
        ml.load_paysim_model.cache_clear()
        ml.load_unified_model.cache_clear()
        status = client.get('/api/v1/model/status', headers=auth)
        assert status.status_code == 200
        assert 'requires scikit-learn' in status.json()['artifact_errors']['paysim']
        response = client.post('/api/v1/paysim/predict', json={'features': {
            'type': 'TRANSFER', 'amount': 5, 'oldbalanceOrg': 5, 'oldbalanceDest': 0,
        }}, headers=auth)
        assert response.status_code == 503
    ml.load_paysim_model.cache_clear()
    ml.load_unified_model.cache_clear()


def test_paysim_feature_contract_excludes_post_event_and_identity_fields():
    forbidden = {'isFraud', 'isFlaggedFraud', 'newbalanceOrig', 'newbalanceDest', 'nameOrig', 'nameDest', 'step'}
    assert not forbidden.intersection(FEATURES)
    assert FEATURES == ['type', 'amount', 'oldbalanceOrg', 'oldbalanceDest']


def test_paysim_v2_features_are_row_local_and_pre_transaction_only():
    rows = pd.DataFrame({'type': ['TRANSFER', 'CASH_OUT'], 'amount': [50., 8.],
                         'oldbalanceOrg': [100., 0.], 'oldbalanceDest': [0., 20.],
                         'isFraud': [1, 0], 'newbalanceOrig': [50., 0.]})
    all_features, numeric, categorical = feature_schema('pre_transaction_v2')
    result = decision_time_features(rows, 'pre_transaction_v2')
    assert list(result) == all_features
    assert not {'isFraud', 'newbalanceOrig', 'newbalanceDest', 'nameOrig', 'nameDest', 'step',
                'origin_balance_after_requested_amount'}.intersection(result)
    assert result.loc[0, 'amount_to_origin_balance'] == 0.5
    assert result.loc[1, 'amount_to_origin_balance'] == 8.0
    assert result.loc[0, 'destination_balance_is_zero'] == 1.0
    assert 'amount_to_origin_balance' in numeric and categorical == ['type']


def test_paysim_v2_inference_derives_only_allowlisted_features(monkeypatch):
    import app.ml as ml

    class Pipeline:
        def predict_proba(self, frame):
            assert list(frame) == list(ml.PAYSIM_V2_FEATURES)
            assert frame.loc[0, 'amount_to_origin_balance'] == 0.5
            assert frame.loc[0, 'amount_to_destination_balance'] == 50.0
            assert frame.loc[0, 'destination_balance_is_zero'] == 1.0
            assert 'origin_balance_after_requested_amount' not in frame
            # sklearn predict_proba returns an indexable 2-D array.
            return np.array([[.2, .8]])

    artifact = {'features': list(ml.PAYSIM_V2_FEATURES), 'pipeline': Pipeline(), 'threshold': .7,
                'model_version': 'unit-v4'}
    monkeypatch.setattr(ml, 'load_paysim_model', lambda: artifact)
    monkeypatch.setattr(ml, '_local_xgboost_explanation', lambda *_: [])
    response = ml.predict_paysim({'type': 'TRANSFER', 'amount': 50., 'oldbalanceOrg': 100., 'oldbalanceDest': 0.})
    assert response['flagged'] is True
    with pytest.raises(ValueError, match='unsupported fields'):
        ml.predict_paysim({'type': 'TRANSFER', 'amount': 50., 'oldbalanceOrg': 100., 'oldbalanceDest': 0.,
                           'origin_balance_after_requested_amount': 50.})


def test_paysim_loader_retains_original_label_only_for_training(tmp_path):
    source = tmp_path / 'paysim.csv'
    pd.DataFrame([{
        'step': 1, 'type': 'TRANSFER', 'amount': 10, 'oldbalanceOrg': 20,
        'oldbalanceDest': 0, 'isFraud': 0, 'isFlaggedFraud': 0,
        'newbalanceOrig': 10, 'newbalanceDest': 10, 'nameOrig': 'C1', 'nameDest': 'M1',
    }]).to_csv(source, index=False)
    loaded = read_paysim(source, max_rows=None)
    assert 'isFraud' in loaded.columns
    assert 'isFlaggedFraud' not in loaded.columns
    assert set(loaded.columns) == {'step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFraud'}


def test_paysim_loader_rejects_missing_required_columns(tmp_path):
    source = tmp_path / 'broken.csv'
    pd.DataFrame([{'step': 1, 'amount': 5, 'isFraud': 0}]).to_csv(source, index=False)
    with pytest.raises(ValueError, match='Missing required PaySim columns'):
        read_paysim(source, max_rows=None)
