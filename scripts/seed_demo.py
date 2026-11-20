"""Seed a clean demo database through the public API.

Runs the six demo scenarios, then adds one analyst comment and one ML
assessment to the highest-risk case, so a fresh dashboard has data to show.
Usage (API already running, .env configured):

    python scripts/seed_demo.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8000'
SCENARIOS = ('normal', 'new_recipient', 'velocity', 'fan_in', 'chain', 'cycle')


def api_key() -> str:
    for line in (ROOT / '.env').read_text(encoding='utf-8-sig').splitlines():
        if line.startswith('IZQAR_API_KEY='):
            return line.split('=', 1)[1].strip()
    raise SystemExit('IZQAR_API_KEY is missing from .env')


def call(method: str, path: str, key: str, payload: dict | None = None):
    request = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'X-API-Key': key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as error:
        raise SystemExit(f'{method} {path} -> {error.code}: {error.read().decode()[:400]}')
    return json.loads(body) if body else None


def main() -> int:
    key = api_key()
    for scenario in SCENARIOS:
        rows = call('POST', '/api/v1/demo/simulate', key, {'scenario': scenario})
        print(f'{scenario}: {len(rows)} transaction(s)')
    queue = call('GET', '/api/v1/investigations?status=all&risk_level=all&search=&sort=highest_risk&limit=50', key)
    target = queue['items'][0]
    call('POST', f"/api/v1/investigations/{target['id']}/comments", key,
         {'comment': 'Проверено на демо-стенде: запрошены документы по источнику средств.'})
    call('POST', f"/api/v1/investigations/{target['id']}/ml-assessments", key,
         {'model': 'paysim', 'features': {'type': 'TRANSFER', 'amount': 181000.0,
                                          'oldbalanceOrg': 181000.0, 'oldbalanceDest': 0.0}})
    detail = call('GET', f"/api/v1/investigations/{target['id']}", key)
    print('case', target['id'], 'events', detail['events_total'],
          'assessments', len(detail['ml_assessments']))
    print('demo seed complete')
    return 0


if __name__ == '__main__':
    sys.exit(main())
