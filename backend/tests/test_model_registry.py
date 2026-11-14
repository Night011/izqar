"""Registry reads metadata without executing serialized model code."""
import json
from pathlib import Path
import sys
import zipfile

import pytest

from app.model_registry import list_registry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from import_experiment_report import import_report  # noqa: E402


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
    assert active['state'] == 'installed' and active['lifecycle'] == 'active' and len(active['components']) == 2
    assert active['test_metrics']['f1'] == 0.2
    assert next(entry for entry in result['entries'] if entry['active_for'] == 'ibm_aml')['state'] == 'invalid'
    candidate = next(entry for entry in result['entries'] if entry['manifest'] == 'trial.metrics.json')
    assert candidate['state'] == 'report_only' and candidate['lifecycle'] == 'candidate'


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
    assert response.json()['registry_version'] == 2


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


def test_registry_marks_nonactive_final_reports_as_archived(tmp_path):
    root = tmp_path / 'models'; root.mkdir()
    report = {'format_version': 2, 'model_kind': 'paysim', 'model_version': 'final-but-not-promoted',
              'features': ['amount'], 'threshold': .5, 'experiment': {'mode': 'final'}}
    manifest = root / 'former.metrics.json'
    manifest.write_text(json.dumps(report), encoding='utf-8')
    entry = list_registry({}, root)['entries'][0]
    assert entry['state'] == 'report_only' and entry['lifecycle'] == 'archived'


def test_imported_validation_report_is_candidate_without_model_payload(tmp_path):
    report = {'format_version': 2, 'model_kind': 'paysim', 'model_version': 'candidate',
              'features': ['amount'], 'threshold': .5, 'run_mode': 'experiment',
              'test_metrics': None, 'baseline_test_metrics': None,
              'experiment': {'mode': 'experiment', 'experiment_id': 'unit'}}
    run = {'mode': 'experiment', 'test_evaluation_performed': False,
           'expected_revision': 'a' * 40, 'actual_revision': 'a' * 40}
    archive = tmp_path / 'candidate.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr('candidate.metrics.json', json.dumps(report))
        zipped.writestr('candidate_run.json', json.dumps(run))
    report_path, run_path = import_report(archive, tmp_path / 'models' / 'experiments')
    assert report_path.is_file() and run_path.is_file()
    entry = list_registry({}, tmp_path / 'models')['entries'][0]
    assert entry['state'] == 'report_only' and entry['lifecycle'] == 'candidate'
    with pytest.raises(FileExistsError):
        import_report(archive, tmp_path / 'models' / 'experiments')
