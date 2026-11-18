"""The browser-facing IBM graph endpoint must not disclose account identifiers."""
import json


def test_ibm_graph_report_endpoint_returns_identifier_free_aggregate(client, auth, tmp_path, monkeypatch):
    report = {
        'rows_analyzed': 250,
        'edge_count': 177,
        'limits': {'max_rows': 500, 'rapid_seconds': 900, 'min_degree': 3, 'max_results': 50},
        'fan_in': [{'account': 'synthetic-account-should-not-leak'}],
        'fan_out': [],
        'rapid_transfer_chains': [{'path': ['secret-A', 'secret-B', 'secret-C']}],
        'directed_three_node_cycles': [['secret-A', 'secret-B', 'secret-C', 'secret-A']],
        'warning': 'Structural signals are investigation leads, not proof.',
    }
    path = tmp_path / 'ibm_graph_report.json'
    path.write_text(json.dumps(report), encoding='utf-8')
    monkeypatch.setattr('app.main.IBM_AML_GRAPH_REPORT_PATH', path)

    response = client.get('/api/v1/graph/ibm-aml-report', headers=auth)

    assert response.status_code == 200
    assert response.json() == {
        'available': True, 'source': 'IBM AML HI-Small synthetic transfers',
        'rows_analyzed': 250, 'edge_count': 177,
        'limits': report['limits'],
        'signal_counts': {'fan_in': 1, 'fan_out': 0, 'rapid_transfer_chains': 1,
                          'directed_three_node_cycles': 1},
        'warning': report['warning'],
    }
    assert 'secret-A' not in response.text
    assert 'synthetic-account-should-not-leak' not in response.text


def test_ibm_graph_report_endpoint_is_honest_when_missing(client, auth, tmp_path, monkeypatch):
    monkeypatch.setattr('app.main.IBM_AML_GRAPH_REPORT_PATH', tmp_path / 'missing.json')
    response = client.get('/api/v1/graph/ibm-aml-report', headers=auth)
    assert response.status_code == 200
    assert response.json() == {'available': False,
                               'warning': 'IBM AML graph report has not been imported yet.'}
