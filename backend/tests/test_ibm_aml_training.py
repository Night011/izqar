"""Small synthetic fixtures exercise IBM AML chronology and schema, not real quality."""
import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_ibm_aml import EXPECTED_HEADER, FEATURES, validate_file, time_boundaries, load_splits, train  # noqa: E402
from export_portable_model import export  # noqa: E402
from install_model_bundle import install  # noqa: E402


def make_csv(path: Path):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(EXPECTED_HEADER)
        for i in range(400):
            day = i // 50 + 1
            writer.writerow([f'2025/01/{day:02d} {i % 24:02d}:{i % 60:02d}', '001', f'A{i}',
                             '002' if i % 2 else '001', f'B{i}', 100 + i,
                             'EUR' if i % 3 else 'USD', float(i % 30 + 1),
                             'USD', 'Wire' if i % 2 else 'ACH', int(i % 13 == 0)])


def test_ibm_aml_temporal_train_export_and_schema(tmp_path, client, auth, monkeypatch):
    path = tmp_path / 'HI-Small_Trans.csv'
    make_csv(path)
    validate_file(path)
    train_cut, valid_cut, total, train_available = time_boundaries(path)
    assert total == 400 and train_cut < valid_cut
    split, counts = load_splits(path, train_cut, valid_cut, train_available, 100, False)
    assert counts == {'train': 280, 'validation': 60, 'test': 60}
    assert len(split['train'][1]) == 100 and split['test'] is None
    assert set(FEATURES).isdisjoint({'Is Laundering', 'Account', 'Account.1', 'Amount Received'})
    experiment = tmp_path / 'experiment.joblib'
    experimental = train(path, experiment, mode='experiment', n_estimators=8,
                         max_train_rows=100, max_false_alerts_per_10000=None)
    assert experimental['test_metrics'] is None and not experiment.exists()
    final = tmp_path / 'final.joblib'
    report = train(path, final, mode='final', n_estimators=8,
                   max_train_rows=100, max_false_alerts_per_10000=None)
    assert report['test_metrics']['cases'] == 60
    assert report['test_metrics']['precision_at_1000'] >= 0
    assert json.loads(final.with_suffix('.metrics.json').read_text())['test_metrics'] == report['test_metrics']
    bundle = export(final, tmp_path / 'ibm_portable')
    installed = install(bundle, tmp_path / 'installed')
    assert len(installed) == 3
    import app.ml as ml
    sample = split['train'][0].iloc[0].to_dict()
    with monkeypatch.context() as patch:
        patch.setattr(ml, 'IBM_AML_MODEL_PATH', tmp_path / 'installed' / 'ibm_portable.metrics.json')
        ml.load_ibm_aml_model.cache_clear()
        status = client.get('/api/v1/model/status', headers=auth)
        assert status.status_code == 200
        assert status.json()['ibm_aml']['model_version'] == 'ibm-aml-hi-small-xgboost-v1'
        response = client.post('/api/v1/ibm-aml/predict', json={'features': sample}, headers=auth)
        assert response.status_code == 200, response.text
        assert response.json()['source'] == 'ibm_aml_ml'
        assert client.post('/api/v1/ibm-aml/predict',
                           json={'features': {**sample, 'Is Laundering': 1}}, headers=auth).status_code == 422
    ml.load_ibm_aml_model.cache_clear()
    with pytest.raises(FileExistsError):
        train(path, final, mode='final', n_estimators=8)


def test_ibm_aml_rejects_unexpected_header(tmp_path):
    path = tmp_path / 'HI-Small_Trans.csv'
    make_csv(path)
    text = path.read_text(encoding='utf-8').replace('Is Laundering', 'isFraud', 1)
    path.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError, match='header'):
        validate_file(path)
