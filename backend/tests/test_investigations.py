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
