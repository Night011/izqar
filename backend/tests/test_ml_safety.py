"""Tests for schema boundaries and leakage guards; no external datasets required."""
from pathlib import Path
import sys
import pytest

pd = pytest.importorskip('pandas', reason='ML dependency not installed')
pytest.importorskip('sklearn', reason='ML dependency not installed')
pytest.importorskip('xgboost', reason='ML dependency not installed')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_paysim import FEATURES, read_paysim  # noqa: E402


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
        patch.setattr(ml.joblib, 'load', lambda *_: pytest.fail('must reject before unpickling'))
        ml.load_paysim_model.cache_clear()
        status = client.get('/api/v1/model/status', headers=auth)
        assert status.status_code == 200
        assert 'requires scikit-learn' in status.json()['artifact_errors']['paysim']
        response = client.post('/api/v1/paysim/predict', json={'features': {
            'type': 'TRANSFER', 'amount': 5, 'oldbalanceOrg': 5, 'oldbalanceDest': 0,
        }}, headers=auth)
        assert response.status_code == 503
    ml.load_paysim_model.cache_clear()


def test_paysim_feature_contract_excludes_post_event_and_identity_fields():
    forbidden = {'isFraud', 'isFlaggedFraud', 'newbalanceOrig', 'newbalanceDest', 'nameOrig', 'nameDest', 'step'}
    assert not forbidden.intersection(FEATURES)
    assert FEATURES == ['type', 'amount', 'oldbalanceOrg', 'oldbalanceDest']


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
