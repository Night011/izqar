"""Installing a model bundle must never overwrite files or extract arbitrary paths."""
import json
from pathlib import Path
import sys
import zipfile

import pytest
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from install_model_bundle import install  # noqa: E402


def bundle(path, *, model_name='paysim_model_v2.joblib', features=None, extra=None):
    report = {
        'model_kind': 'paysim', 'model_version': 'paysim-xgboost-v2',
        'library_versions': {'scikit_learn': sklearn.__version__},
        'features': features or ['type', 'amount', 'oldbalanceOrg', 'oldbalanceDest'],
        'threshold': 0.8, 'test_metrics': {'cases': 10},
    }
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(model_name, b'test-only-not-a-real-model')
        archive.writestr(model_name.removesuffix('.joblib') + '.metrics.json', json.dumps(report))
        if extra:
            archive.writestr(extra, 'unexpected')


def test_bundle_install_and_no_overwrite(tmp_path):
    source = tmp_path / 'bundle.zip'
    bundle(source)
    destination = tmp_path / 'models'
    installed = install(source, destination)
    assert len(installed) == 2
    assert installed[0].read_bytes() == b'test-only-not-a-real-model'
    with pytest.raises(FileExistsError):
        install(source, destination)
    assert installed[0].read_bytes() == b'test-only-not-a-real-model'


@pytest.mark.parametrize('model_name,features,extra', [
    ('../escape.joblib', None, None),
    ('paysim_model_v2.joblib', ['isFraud'], None),
    ('paysim_model_v2.joblib', None, '../outside.txt'),
])
def test_bundle_rejects_unsafe_contents(tmp_path, model_name, features, extra):
    source = tmp_path / 'unsafe.zip'
    bundle(source, model_name=model_name, features=features, extra=extra)
    with pytest.raises(ValueError):
        install(source, tmp_path / 'models')
    assert not (tmp_path / 'outside.txt').exists()


def test_bundle_checks_digest(tmp_path):
    source = tmp_path / 'bundle.zip'
    bundle(source)
    with pytest.raises(ValueError, match='SHA-256'):
        install(source, tmp_path / 'models', '0' * 64)
