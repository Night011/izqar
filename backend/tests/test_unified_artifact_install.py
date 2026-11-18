"""The Unified installer must reject unreviewed archives before writing artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from install_unified_artifact import install  # noqa: E402


def test_installer_verifies_components_and_refuses_overwrite(tmp_path):
    content = {
        'unified_weights.npz': b'weights',
        'unified_preprocessors.joblib': b'preprocessors',
        'unified.metrics.json': json.dumps({'format_version': 1, 'model_kind': 'izqar_unified_mtl',
                                             'serialization': 'numpy_shared_backbone_v1'}).encode(),
        'unified_checkpoint.pt': b'checkpoint',
    }
    expected = {name: hashlib.sha256(value).hexdigest() for name, value in content.items()}
    archive = tmp_path / 'unified.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        for name, value in content.items():
            bundle.writestr(name, value)
    destination = tmp_path / 'models'
    installed = install(archive, destination, expected)
    assert {path.name for path in installed} == set(content)
    assert (destination / 'unified.metrics.json').read_bytes() == content['unified.metrics.json']
    with pytest.raises(FileExistsError):
        install(archive, destination, expected)


def test_installer_rejects_digest_mismatch_before_writing(tmp_path):
    archive = tmp_path / 'unified.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('unified_weights.npz', b'not-reviewed')
    with pytest.raises(ValueError, match='SHA-256'):
        install(archive, tmp_path / 'models', {'unified_weights.npz': '0' * 64})
    assert not (tmp_path / 'models').exists()
