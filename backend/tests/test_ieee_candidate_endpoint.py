"""Endpoint contract for the experimental IEEE-CIS candidate metrics."""
import json


def test_ieee_candidate_404_when_absent(client, auth, monkeypatch):
    monkeypatch.setattr('app.ml.load_ieee_candidate_metrics', lambda: None)
    assert client.get('/api/v1/model/ieee-candidate', headers=auth).status_code == 404


def test_ieee_candidate_200_with_metrics(client, auth, monkeypatch, tmp_path):
    metrics = {'model_version': 'ieee-cis-xgboost-v3-candidate',
               'test_metrics': {'average_precision_pr_auc': 0.507407},
               'validation_metrics': {'average_precision_pr_auc': 0.565865},
               'threshold': 0.828, 'split_method': 'test-split',
               'warning': 'research'}
    path = tmp_path / 'ieee_v3_candidate.metrics.json'
    path.write_text(json.dumps(metrics), encoding='utf-8')
    monkeypatch.setenv('IZQAR_IEEE_CANDIDATE_METRICS', str(path))
    response = client.get('/api/v1/model/ieee-candidate', headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body['model_version'] == 'ieee-cis-xgboost-v3-candidate'
    assert body['experimental'] is True
    assert body['test_metrics']['average_precision_pr_auc'] == 0.507407


def test_ieee_candidate_requires_auth(client):
    assert client.get('/api/v1/model/ieee-candidate').status_code == 401
