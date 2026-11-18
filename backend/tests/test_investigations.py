"""Investigation actions must persist and be independently retrievable."""


def test_case_queue_decision_comments_and_related_history(client, auth):
    first = client.post('/api/v1/demo/simulate', json={'scenario': 'fan_in'}, headers=auth)
    assert first.status_code == 200
    rows = first.json()
    case = next(row for row in rows if row['risk_level'] == 'high')
    queue = client.get('/api/v1/investigations?sort=highest_risk', headers=auth)
    assert queue.status_code == 200
    assert queue.json()['total'] >= 1
    assert any(row['id'] == case['id'] for row in queue.json()['items'])
    assert client.get('/api/v1/investigations', headers={'X-API-Key': 'wrong'}).status_code == 401

    update = client.patch(f"/api/v1/transactions/{case['id']}/review", headers=auth,
                          json={'review_state': 'step_up', 'comment': 'Запросить подтверждение клиента'})
    assert update.status_code == 200, update.text
    assert update.json()['review_state'] == 'step_up'
    comment = client.post(f"/api/v1/investigations/{case['id']}/comments", headers=auth,
                          json={'comment': '   Проверить связанные переводы   '})
    assert comment.status_code == 200, comment.text
    assert comment.json()['comment'] == 'Проверить связанные переводы'
    assert comment.json()['actor_source'] == 'local_demo_api_key'
    detail = client.get(f"/api/v1/investigations/{case['id']}", headers=auth)
    assert detail.status_code == 200
    body = detail.json()
    assert body['transaction']['review_state'] == 'step_up'
    assert body['events_total'] == 2
    assert [(event['action'], event['to_state']) for event in body['events']] == [
        ('decision', 'step_up'), ('comment', 'step_up')]
    assert body['events'][0]['from_state'] == 'pending'
    assert all(event['created_at'] for event in body['events'])
    assert any(row['sender'] == 'MULE-001' or row['recipient'] == 'MULE-001'
               for row in body['related_transactions'])
    assert client.get('/api/v1/investigations?status=step_up', headers=auth).json()['total'] == 1
    assert client.get('/api/v1/investigations?search=MULE-001', headers=auth).json()['total'] >= 1
    assert client.get('/api/v1/investigations?search=%25', headers=auth).json()['total'] == 0


def test_no_unlogged_or_invalid_case_changes(client, auth):
    row = client.post('/api/v1/demo/simulate', json={'scenario': 'new_recipient'}, headers=auth).json()[0]
    path = f"/api/v1/transactions/{row['id']}/review"
    assert client.patch(path, headers=auth, json={'review_state': 'invented'}).status_code == 422
    assert client.patch(path, headers=auth, json={'review_state': 'in_review', 'comment': '  '}).status_code == 422
    assert client.post(f"/api/v1/investigations/{row['id']}/comments", headers=auth,
                       json={'comment': '  '}).status_code == 422
    assert client.get(f"/api/v1/investigations/{row['id']}", headers=auth).json()['events'] == []
    assert client.patch('/api/v1/transactions/missing/review', headers=auth,
                        json={'review_state': 'allowed'}).status_code == 404
    assert client.post('/api/v1/investigations/missing/comments', headers=auth,
                       json={'comment': 'x'}).status_code == 404
    assert client.get('/api/v1/investigations/missing', headers=auth).status_code == 404

    response = client.patch(path, headers=auth, json={'review_state': 'allowed'})
    assert response.status_code == 200
    detail = client.get(f"/api/v1/investigations/{row['id']}", headers=auth).json()
    assert len(detail['events']) == 1
    assert detail['events'][0]['to_state'] == 'allowed'
    assert detail['transaction']['decision'] == 'review'  # original rule suggestion is retained


def test_case_can_record_only_server_produced_ml_assessment(client, auth, monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, 'predict_paysim', lambda _: {
        'risk_score': 77.7, 'threshold': .8, 'flagged': False, 'source': 'paysim_ml',
        'model_version': 'test-model', 'score_kind': 'uncalibrated_ml_score', 'warning': 'test warning',
    })
    row = client.post('/api/v1/demo/simulate', json={'scenario': 'new_recipient'}, headers=auth).json()[0]
    path = f"/api/v1/investigations/{row['id']}/ml-assessments"
    response = client.post(path, headers=auth, json={'model': 'paysim', 'features': {
        'type': 'TRANSFER', 'amount': 150000, 'oldbalanceOrg': 180000, 'oldbalanceDest': 20000,
    }})
    assert response.status_code == 200, response.text
    assessment = response.json()
    assert assessment['source'] == 'paysim_ml'
    assert assessment['score_kind'] == 'uncalibrated_ml_score'
    assert assessment['feature_names'] == ['amount', 'oldbalanceDest', 'oldbalanceOrg', 'type']
    # Dataset labels are rejected by the actual ML schema and never persisted.
    invalid = client.post(path, headers=auth, json={'model': 'paysim', 'features': {
        'type': 'TRANSFER', 'amount': 1, 'oldbalanceOrg': 1, 'oldbalanceDest': 1, 'isFraud': 1,
    }})
    assert invalid.status_code == 422
    detail = client.get(f"/api/v1/investigations/{row['id']}", headers=auth).json()
    assert len(detail['ml_assessments']) == 1
    assert detail['events'][-1]['action'] == 'ml_assessment'
