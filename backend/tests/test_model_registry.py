"""Registry reads metadata without executing serialized model code."""
import json
from pathlib import Path

from app.model_registry import list_registry


def test_registry_active_archive_and_missing_component(tmp_path):
    root = tmp_path / 'models'
    root.mkdir()
    report = {'format_version': 3, 'serialization': 'portable_xgboost_v1',
              'model_kind': 'paysim', 'model_version': 'unit-only',
              'features': ['amount'], 'threshold': 0.8, 'test_metrics': {'f1': 0.2},
              'data_files': [{'filename': 'fixture.csv', 'sha256': 'fixture-only'}]}
    manifest = root / 'active.metrics.json'
    manifest.write_text(json.dumps(report), encoding='utf-8')
    (root / 'active.preprocessor.joblib').write_bytes(b'not-a-pickle-and-never-loaded')
    (root / 'active.xgboost.json').write_text('{}', encoding='utf-8')
    archived = root / 'experiments'
    archived.mkdir()
    (archived / 'trial.metrics.json').write_text(json.dumps(report), encoding='utf-8')
    result = list_registry({'paysim': manifest, 'ibm_aml': root / 'absent.metrics.json'}, root)
    active = next(entry for entry in result['entries'] if entry['active_for'] == 'paysim')
    assert active['state'] == 'installed' and len(active['components']) == 2
    assert active['test_metrics']['f1'] == 0.2
    assert next(entry for entry in result['entries'] if entry['active_for'] == 'ibm_aml')['state'] == 'invalid'
    assert next(entry for entry in result['entries'] if entry['manifest'] == 'trial.metrics.json')['state'] == 'report_only'


def test_registry_rejects_leaky_schema_and_protects_api(client, auth, tmp_path):
    root = tmp_path / 'models'
    root.mkdir()
    manifest = root / 'bad.metrics.json'
    manifest.write_text(json.dumps({'format_version': 2, 'model_kind': 'paysim',
                                    'features': ['isFraud'], 'threshold': 0.5}), encoding='utf-8')
    entry = list_registry({'paysim': manifest}, root)['entries'][0]
    assert entry['state'] == 'invalid' and 'forbidden' in entry['error']
    assert client.get('/api/v1/model/registry').status_code == 401
    response = client.get('/api/v1/model/registry', headers=auth)
    assert response.status_code == 200
    assert response.json()['registry_version'] == 1


def test_registry_accepts_unified_manifest_only_with_four_clean_task_schemas(tmp_path):
    root = tmp_path / 'models'; root.mkdir()
    tasks = {task: {'feature_schema': {'all': ['amount'], 'numerical': ['amount'], 'categorical': []}}
             for task in ('paysim', 'ieee_cis', 'baf_base', 'ibm_aml')}
    report = {'format_version': 1, 'serialization': 'numpy_shared_backbone_v1',
              'model_kind': 'izqar_unified_mtl', 'model_version': 'unified-test',
              'task_metadata': tasks, 'tasks': {task: {'threshold': .5} for task in tasks}}
    manifest = root / 'unified.metrics.json'; manifest.write_text(json.dumps(report), encoding='utf-8')
    (root / 'unified_preprocessors.joblib').write_bytes(b'trusted-only-placeholder')
    (root / 'unified_weights.npz').write_bytes(b'not-loaded-by-registry')
    result = list_registry({'izqar_unified_mtl': manifest}, root)['entries'][0]
    assert result['state'] == 'installed' and result['experimental'] is True
    assert result['task_schemas']['paysim'] == ['amount']
    tasks['paysim']['feature_schema']['all'] = ['nameOrig']
    manifest.write_text(json.dumps(report), encoding='utf-8')
    assert list_registry({'izqar_unified_mtl': manifest}, root)['entries'][0]['state'] == 'invalid'
