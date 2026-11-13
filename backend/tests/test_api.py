def test_health_and_access_control(client, auth):
    assert client.get('/health').status_code == 200
    assert client.get('/api/v1/transactions').status_code == 401
    assert client.get('/api/v1/transactions', headers={'X-API-Key': 'wrong'}).status_code == 401
    assert client.get('/api/v1/transactions', headers=auth).status_code == 200


def test_basic_transfer_and_metrics(client, auth):
    payload = {'sender': 'TEST-001', 'recipient': 'TEST-002', 'amount': 2500}
    created = client.post('/api/v1/transfers', json=payload, headers=auth)
    assert created.status_code == 200, created.text
    row = created.json()
    assert row['risk_level'] == 'low'
    assert row['decision'] == 'allow'
    assert row['indicators']['score_type'] == 'rule_based_demo_not_probability'
    assert client.get('/api/v1/metrics', headers=auth).json()['transactions'] == 1


def test_high_risk_review(client, auth):
    row = client.post('/api/v1/demo/simulate', json={'scenario': 'new_recipient'}, headers=auth).json()[0]
    assert row['risk_score'] >= 70
    assert row['decision'] == 'review'
    update = client.patch(f"/api/v1/transactions/{row['id']}/review", json={'review_state': 'confirmed_fraud'}, headers=auth)
    assert update.status_code == 200
    assert update.json()['review_state'] == 'confirmed_fraud'


def test_graph_fan_in_and_velocity(client, auth):
    r = client.post('/api/v1/demo/simulate', json={'scenario': 'fan_in'}, headers=auth)
    assert r.status_code == 200
    graph = client.get('/api/v1/graph', headers=auth).json()
    assert 'MULE-001' in graph['indicators']['high_fan_in_accounts']
    assert len(graph['edges']) == 6
    r2 = client.post('/api/v1/demo/simulate', json={'scenario': 'velocity'}, headers=auth)
    assert r2.json()[-1]['risk_score'] > r2.json()[0]['risk_score']


def test_validation_and_model_unavailable(client, auth):
    bad = client.post('/api/v1/transfers', json={'sender': 'A-1', 'recipient': 'A-1', 'amount': 200}, headers=auth)
    assert bad.status_code == 422
    bad = client.post('/api/v1/transfers', json={'sender': 'X-1', 'recipient': 'Y-1', 'amount': -10}, headers=auth)
    assert bad.status_code == 422
    missing = client.post('/api/v1/ieee/predict', json={'features': {'TransactionAmt': 5}}, headers=auth)
    assert missing.status_code == 503


def test_autonomous_demo_page(client):
    response = client.get('/demo/')
    assert response.status_code == 200
    assert 'IZQAR' in response.text
