"""Endpoint contract for the experimental BAF candidate metrics."""
import json


def test_baf_candidate_404_when_absent(client, auth, monkeypatch):
    monkeypatch.setattr('app.ml.load_baf_candidate_metrics', lambda: None)
    assert client.get('/api/v1/model/baf-candidate', headers=auth).status_code == 404


def test_baf_candidate_200_with_metrics(client, auth, monkeypatch, tmp_path):
    metrics = {'model_version': 'baf-research-folds',
               'mean_metrics': {'average_precision_pr_auc': 0.156467, 'f1': 0.226626},
               'folds': [{'valid_month': 3}, {'valid_month': 4}, {'valid_month': 5}],
               'split_method': 'expanding month folds', 'note': 'test months untouched'}
    path = tmp_path / 'baf_research_v1.metrics.json'
    path.write_text(json.dumps(metrics), encoding='utf-8')
    monkeypatch.setenv('IZQAR_BAF_CANDIDATE_METRICS', str(path))
    response = client.get('/api/v1/model/baf-candidate', headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body['experimental'] is True
    assert body['mean_metrics']['average_precision_pr_auc'] == 0.156467
    assert len(body['folds']) == 3


def test_baf_candidate_requires_auth(client):
    assert client.get('/api/v1/model/baf-candidate').status_code == 401
