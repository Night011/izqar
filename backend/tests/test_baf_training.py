"""BAF fixture verifies temporal and leakage boundaries; its scores are not research metrics."""
import json
from pathlib import Path
import sys

import pytest

pd = pytest.importorskip('pandas')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_baf import CATEGORICAL, FEATURES, NUMERIC, read_base, train  # noqa: E402
from export_portable_model import export as export_portable_model  # noqa: E402
from install_model_bundle import install as install_model_bundle  # noqa: E402


def make_base(path: Path):
    n = 400
    data = {'month': [i // 50 for i in range(n)],
            'fraud_bool': [int(i % 10 == 0) for i in range(n)],
            'device_fraud_count': [int(i % 10 == 0) for i in range(n)],
            'days_since_request': [int(i % 10 == 0) for i in range(n)]}
    for name in NUMERIC:
        data[name] = [float(i % 15 + 1) for i in range(n)]
    for name in CATEGORICAL:
        data[name] = ['a' if i % 2 else 'b' for i in range(n)]
    data['prev_address_months_count'][0] = -1
    pd.DataFrame(data).to_csv(path, index=False)


def test_baf_base_train_and_final_holdout(tmp_path, client, auth, monkeypatch):
    path = tmp_path / 'Base.csv'
    make_base(path)
    data = read_base(path)
    assert pd.isna(data.loc[0, 'prev_address_months_count'])
    experiment = tmp_path / 'experiment.joblib'
    report = train(path, experiment, n_estimators=8, max_train_rows=100, mode='experiment',
                   max_false_alerts_per_10000=None)
    assert not experiment.exists()
    assert report['test_metrics'] is None
    assert report['run_mode'] == 'experiment'
    assert set(report['validation_operating_profiles']) == {'monitoring', 'analyst_review', 'high_confidence'}
    weighted = tmp_path / 'weighted.joblib'
    weighted_report = train(path, weighted, n_estimators=8, max_train_rows=100, mode='experiment',
                            max_false_alerts_per_10000=None, positive_weight_multiplier=1.25)
    assert weighted_report['experiment']['hyperparameters']['positive_weight_multiplier'] == 1.25
    assert report['split_boundaries'] == {'train_months': [0, 1, 2, 3, 4, 5],
                                          'validation_month': 6, 'test_month': 7}
    assert set(FEATURES).isdisjoint({'fraud_bool', 'month', 'device_fraud_count', 'days_since_request'})
    final = tmp_path / 'final.joblib'
    final_report = train(path, final, n_estimators=8, mode='final',
                         max_false_alerts_per_10000=None)
    assert final.is_file()
    assert final_report['run_mode'] == 'final'
    assert final_report['test_metrics']['cases'] == 50
    saved = json.loads(final.with_suffix('.metrics.json').read_text(encoding='utf-8'))
    assert saved['test_metrics'] == final_report['test_metrics']
    with pytest.raises(FileExistsError):
        train(path, final, n_estimators=8, mode='final')
    import app.ml as ml
    sample = pd.read_csv(path).iloc[10][FEATURES].to_dict()
    with monkeypatch.context() as patch:
        patch.setattr(ml, 'BAF_MODEL_PATH', final)
        ml.load_baf_model.cache_clear()
        status = client.get('/api/v1/model/status', headers=auth)
        assert status.status_code == 200
        assert status.json()['baf_base']['model_version'] == 'baf-base-xgboost-v1'
        response = client.post('/api/v1/baf/predict', json={'features': sample}, headers=auth)
        assert response.status_code == 200, response.text
        assert response.json()['source'] == 'baf_base_ml'
        assert client.post('/api/v1/baf/predict', json={'features': {**sample, 'fraud_bool': 1}}, headers=auth).status_code == 422
        bundle = export_portable_model(final, tmp_path / 'baf_portable')
        install_model_bundle(bundle, tmp_path / 'installed')
        patch.setattr(ml, 'BAF_MODEL_PATH', tmp_path / 'installed' / 'baf_portable.metrics.json')
        ml.load_baf_model.cache_clear()
        portable_response = client.post('/api/v1/baf/predict', json={'features': sample}, headers=auth)
        assert portable_response.status_code == 200, portable_response.text
        assert portable_response.json()['risk_score'] == response.json()['risk_score']
    ml.load_baf_model.cache_clear()


def test_baf_rejects_wrong_variant_and_missing_schema(tmp_path):
    path = tmp_path / 'TypeI.csv'
    make_base(path)
    with pytest.raises(ValueError, match='Base.csv'):
        read_base(path)
    base = tmp_path / 'Base.csv'
    pd.DataFrame({'month': [0], 'fraud_bool': [1]}).to_csv(base, index=False)
    with pytest.raises(ValueError, match='Missing BAF Base columns'):
        read_base(base)
