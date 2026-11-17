"""IBM offline shadow bundle fixtures: no Kaggle CSV, no sealed test and no API promotion."""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ibm_shadow_bundle as shadow  # noqa: E402


def fitted_candidate():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        col: rng.uniform(0, 200, size=48).astype('float32')
        for col in shadow.BASE_NUMERIC + shadow.HISTORY_NUMERIC
    })
    frame['same_currency'] = 1.0
    frame['same_bank'] = 0.0
    frame['receiving_currency'] = ['USD', 'EUR'] * 24
    frame['payment_currency'] = ['USD', 'EUR'] * 24
    frame['payment_format'] = ['Wire', 'ACH'] * 24
    labels = np.array([0, 1] * 24, dtype='int8')
    pre = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median'))]), list(shadow.BASE_NUMERIC + shadow.HISTORY_NUMERIC)),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), list(shadow.CATEGORICAL)),
    ])
    model = XGBClassifier(n_estimators=8, max_depth=2, random_state=42, n_jobs=1,
                          objective='binary:logistic', tree_method='hist')
    pipeline = Pipeline([('preprocessing', pre), ('model', model)])
    pipeline.fit(frame[list(shadow.EXPECTED_FEATURES)], labels)
    report = {
        'format_version': 2, 'model_kind': 'ibm_aml', 'model_version': 'ibm-aml-causal-shadow-test',
        'run_mode': 'experiment', 'experiment': {'mode': 'experiment', 'experiment_id': 'fixture',
                                                 'code_revision': 'a' * 40,
                                                 'hyperparameters': {'feature_set': 'causal_history_v1'}},
        'data_files': [{'filename': 'HI-Small_Trans.csv', 'sha256': 'b' * 64, 'size_bytes': 1000}],
        'library_versions': {'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__},
        'features': list(shadow.EXPECTED_FEATURES),
        'numerical_features': list(shadow.BASE_NUMERIC + shadow.HISTORY_NUMERIC),
        'categorical_features': list(shadow.CATEGORICAL),
        'threshold': 0.5, 'validation_metrics': {'average_precision_pr_auc': 0.12, 'cases': 48},
        'test_metrics': None, 'baseline_test_metrics': None,
    }
    return pipeline, report


def test_ibm_shadow_bundle_roundtrip_and_registry_candidate(tmp_path):
    pipeline, report = fitted_candidate()
    archive = tmp_path / 'ibm_causal_shadow.zip'
    summary = shadow.export_shadow_bundle(pipeline, report, archive)
    assert archive.is_file()
    assert summary['synthetic_smoke_max_delta'] <= 1e-6
    verified = shadow.verify_shadow_bundle(archive, summary['sha256'])
    assert verified['stem'] == archive.stem
    output = tmp_path / 'models' / 'experiments' / 'ibm_shadow'
    installed = shadow.install_shadow_bundle(archive, output, summary['sha256'])
    assert len(installed['installed_files']) == 4
    smoke = shadow.trusted_smoke_installed(output, verified['stem'])
    assert smoke['model_load'] == 'passed' and smoke['predict'] == 'passed'
    assert 0 <= smoke['uncalibrated_score'] <= 1
    with pytest.raises(FileExistsError):
        shadow.install_shadow_bundle(archive, output, summary['sha256'])


def test_ibm_shadow_is_only_candidate_in_registry(tmp_path):
    from app.model_registry import list_registry
    pipeline, report = fitted_candidate()
    archive = tmp_path / 'shadow.zip'
    digest = shadow.export_shadow_bundle(pipeline, report, archive)['sha256']
    shadow.install_shadow_bundle(archive, tmp_path / 'models' / 'experiments' / 'shadow', digest)
    entries = list_registry({}, tmp_path / 'models')['entries']
    assert len(entries) == 1
    assert entries[0]['state'] == 'installed'
    assert entries[0]['lifecycle'] == 'candidate'
    assert entries[0]['active_for'] is None


def test_shadow_rejects_missing_pin_and_different_hash(tmp_path):
    pipeline, report = fitted_candidate()
    archive = tmp_path / 'shadow.zip'
    shadow.export_shadow_bundle(pipeline, report, archive)
    with pytest.raises(ValueError, match='SHA-256'):
        shadow.verify_shadow_bundle(archive, '0' * 64)
    with pytest.raises(TypeError):
        shadow.install_shadow_bundle(archive, tmp_path / 'experiments' / 'test')


def test_shadow_rejects_final_metrics_and_forbidden_features(tmp_path):
    pipeline, report = fitted_candidate()
    report['test_metrics'] = {'f1': 0.9}
    with pytest.raises(ValueError, match='final-test'):
        shadow.export_shadow_bundle(pipeline, report, tmp_path / 'test.zip')
    report['test_metrics'] = None
    report['features'].append('Account')
    with pytest.raises(ValueError, match='schema'):
        shadow.export_shadow_bundle(pipeline, report, tmp_path / 'test.zip')


def test_shadow_rejects_zip_bomb_before_extraction(tmp_path, monkeypatch):
    pipeline, report = fitted_candidate()
    archive = tmp_path / 'shadow.zip'
    shadow.export_shadow_bundle(pipeline, report, archive)
    monkeypatch.setattr(shadow, 'MAX_TOTAL_UNCOMPRESSED', 10)
    with pytest.raises(ValueError, match='uncompressed'):
        shadow.verify_shadow_bundle(archive)
    assert not (tmp_path / 'models').exists()


def test_shadow_rejects_archive_path_traversal(tmp_path):
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr('../escape.metrics.json', '{}')
        zipped.writestr('test.shadow.json', '{}')
        zipped.writestr('test.xgboost.json', '{}')
        zipped.writestr('test.preprocessor.joblib', b'unsafe')
    with pytest.raises(ValueError, match='Unsafe'):
        shadow.verify_shadow_bundle(archive)


def test_experiment_ibm_shadow_not_accepted_as_api_model(tmp_path, monkeypatch):
    """Crucial: fail *before* the candidate preprocessor is unpickled."""
    from app import ml
    candidate = tmp_path / 'shadow.metrics.json'
    pipeline, report = fitted_candidate()
    candidate.write_text(json.dumps(report), encoding='utf-8')
    with monkeypatch.context() as patch:
        patch.setattr(ml, 'IBM_AML_MODEL_PATH', candidate)
        patch.setattr(ml.joblib, 'load', lambda *_: pytest.fail('never deserialize experiment into API'))
        ml.load_ibm_aml_model.cache_clear()
        with pytest.raises(RuntimeError, match='experiment-only'):
            ml.load_ibm_aml_model()
    ml.load_ibm_aml_model.cache_clear()


def test_shadow_export_cli_only_for_causal_experiment(tmp_path):
    from train_ibm_aml import train
    with pytest.raises(ValueError, match='Shadow export'):
        train(tmp_path / 'missing.csv', tmp_path / 'result.joblib', mode='final',
              feature_set='static_v1', shadow_bundle=tmp_path / 'shadow.zip')
    with pytest.raises(ValueError, match='Shadow export'):
        train(tmp_path / 'missing.csv', tmp_path / 'result.joblib', mode='experiment',
              feature_set='static_v1', shadow_bundle=tmp_path / 'shadow.zip')


def test_ibm_shadow_notebook_is_revision_pinned_and_cannot_run_final():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / 'notebooks' / 'ibm_causal_shadow_kaggle.ipynb').read_text(encoding='utf-8'))
    content = '\n'.join(''.join(cell.get('source', [])) for cell in notebook['cells'])
    assert notebook['nbformat'] == 4
    assert "EXPECTED_REVISION = 'REPLACE_WITH_40_CHAR_COMMIT_SHA'" in content
    assert "['git', 'checkout', '--detach', EXPECTED_REVISION]" in content
    assert "'--mode', 'experiment'" in content
    assert "'--shadow-bundle', str(bundle)" in content
    assert "assert report['test_metrics'] is None" in content
    assert "assert not output.exists()" in content
    assert "'--mode', 'final'" not in content


def test_shadow_export_requires_revision_dataset_and_causal_hyperparameters(tmp_path):
    pipeline, report = fitted_candidate()
    report['experiment']['code_revision'] = None
    with pytest.raises(ValueError, match='code revision'):
        shadow.export_shadow_bundle(pipeline, report, tmp_path / 'missing_revision.zip')
    report['experiment']['code_revision'] = 'a' * 40
    report['experiment']['hyperparameters']['feature_set'] = 'static_v1'
    with pytest.raises(ValueError, match='causal_history_v1'):
        shadow.export_shadow_bundle(pipeline, report, tmp_path / 'wrong_featureset.zip')
    report['experiment']['hyperparameters']['feature_set'] = 'causal_history_v1'
    report['data_files'][0]['sha256'] = 'not-a-hash'
    with pytest.raises(ValueError, match='SHA-256-pinned'):
        shadow.export_shadow_bundle(pipeline, report, tmp_path / 'missing_data_provenance.zip')


def test_trusted_smoke_rejects_incompatible_sklearn_before_joblib(tmp_path, monkeypatch):
    import joblib
    import sklearn
    pipeline, report = fitted_candidate()
    archive = tmp_path / 'shadow.zip'
    digest = shadow.export_shadow_bundle(pipeline, report, archive)['sha256']
    destination = tmp_path / 'experiments' / 'shadow'
    shadow.install_shadow_bundle(archive, destination, digest)
    with monkeypatch.context() as patch:
        patch.setattr(sklearn, '__version__', '0.0-incompatible')
        patch.setattr(joblib, 'load', lambda *_: pytest.fail('cannot unpickle before version check'))
        with pytest.raises(RuntimeError, match='needs scikit-learn'):
            shadow.trusted_smoke_installed(destination, archive.stem)


def test_trusted_smoke_detects_tampered_component_before_unpickle(tmp_path, monkeypatch):
    """An installed candidate must never be unpickled after file tampering."""
    import joblib
    pipeline, report = fitted_candidate()
    archive = tmp_path / 'offline.zip'
    digest = shadow.export_shadow_bundle(pipeline, report, archive)['sha256']
    target = tmp_path / 'models' / 'experiments' / 'candidate'
    shadow.install_shadow_bundle(archive, target, digest)
    original = target / 'offline.preprocessor.joblib'
    with original.open('ab') as output:
        output.write(b'edited-after-install')
    with monkeypatch.context() as patch:
        patch.setattr(joblib, 'load', lambda *_: pytest.fail('must verify file hashes before unpickling'))
        with pytest.raises(ValueError, match='hash or metadata mismatch'):
            shadow.trusted_smoke_installed(target, archive.stem)


def test_installed_verifier_checks_manifest_and_rejects_symlinks(tmp_path):
    pipeline, report = fitted_candidate()
    archive = tmp_path / 'offline.zip'
    digest = shadow.export_shadow_bundle(pipeline, report, archive)['sha256']
    target = tmp_path / 'experiments' / 'candidate'
    shadow.install_shadow_bundle(archive, target, digest)
    assert shadow.verify_installed_shadow(target, archive.stem)['status'] == 'verified_local_components'
    manifest_path = target / 'offline.shadow.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['serving_allowed'] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest disagrees'):
        shadow.verify_installed_shadow(target, archive.stem)


def test_explicit_installed_verification_never_deserializes(tmp_path, monkeypatch):
    import joblib
    pipeline, report = fitted_candidate()
    archive = tmp_path / 'offline.zip'
    digest = shadow.export_shadow_bundle(pipeline, report, archive)['sha256']
    target = tmp_path / 'experiments' / 'candidate'
    shadow.install_shadow_bundle(archive, target, digest)
    with monkeypatch.context() as patch:
        patch.setattr(joblib, 'load', lambda *_: pytest.fail('metadata check must not deserialize'))
        assert shadow.verify_installed_shadow(target, archive.stem)['status'] == 'verified_local_components'


def test_shadow_rejects_manifest_feature_set_mismatch_after_install(tmp_path):
    pipeline, report = fitted_candidate()
    archive = tmp_path / 'offline.zip'
    digest = shadow.export_shadow_bundle(pipeline, report, archive)['sha256']
    target = tmp_path / 'experiments' / 'candidate'
    shadow.install_shadow_bundle(archive, target, digest)
    manifest_path = target / 'offline.shadow.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['feature_set'] = 'causal_history_bank_v2'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest disagrees'):
        shadow.verify_installed_shadow(target, 'offline')


def test_registry_does_not_mark_configured_shadow_as_active(tmp_path):
    """A misconfigured active path must never display the candidate as serving."""
    from app.model_registry import list_registry
    pipeline, report = fitted_candidate()
    archive = tmp_path / 'offline.zip'
    digest = shadow.export_shadow_bundle(pipeline, report, archive)['sha256']
    root = tmp_path / 'models'
    target = root / 'experiments' / 'candidate'
    shadow.install_shadow_bundle(archive, target, digest)
    entries = list_registry({'ibm_aml': target / 'offline.metrics.json'}, root)['entries']
    assert len(entries) == 1
    assert entries[0]['lifecycle'] == 'candidate'
    assert entries[0]['active_for'] is None
    assert entries[0]['configured_for'] == 'ibm_aml'
    assert 'NOT serving' in entries[0]['error']


def test_shadow_rejects_symlink_zip_entry_before_reading_payload(tmp_path):
    """A ZIP member marked as Unix symlink is never followed or extracted."""
    import stat
    archive = tmp_path / 'unsafe_symlink.zip'
    symbolic = zipfile.ZipInfo('offline.preprocessor.joblib')
    symbolic.create_system = 3
    symbolic.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr(symbolic, 'outside-target')
        zipped.writestr('offline.xgboost.json', '{}')
        zipped.writestr('offline.metrics.json', '{}')
        zipped.writestr('offline.shadow.json', '{}')
    with pytest.raises(ValueError, match='Unsafe'):
        shadow.verify_shadow_bundle(archive)


def test_installed_manifest_must_be_json_object(tmp_path, monkeypatch):
    import joblib
    pipeline, report = fitted_candidate()
    archive = tmp_path / 'offline.zip'
    digest = shadow.export_shadow_bundle(pipeline, report, archive)['sha256']
    target = tmp_path / 'experiments' / 'candidate'
    shadow.install_shadow_bundle(archive, target, digest)
    (target / 'offline.shadow.json').write_text('[]', encoding='utf-8')
    with monkeypatch.context() as patch:
        patch.setattr(joblib, 'load', lambda *_: pytest.fail('must reject malformed manifest before pickle'))
        with pytest.raises(ValueError, match='not a JSON object'):
            shadow.trusted_smoke_installed(target, 'offline')
