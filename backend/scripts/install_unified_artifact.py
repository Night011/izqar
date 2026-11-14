"""Install the reviewed IZQAR Unified Kaggle export without trusting ZIP paths.

This installer accepts only the four components whose hashes were recorded in
the completed Kaggle validation run.  It is intentionally version-specific:
a new training run needs its own reviewed component digests before it can be
installed.  Joblib data is never deserialized by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


MAX_COMPONENT_BYTES = 5 * 1024 * 1024
TRUSTED_COMPONENT_SHA256 = {
    'unified_weights.npz': 'f4c36b169a65471b6b7ecaa11deed32d1cd9a966d6e5f067c07fc3a13c0f23dc',
    'unified_preprocessors.joblib': 'c8275c4ab0de9d9f250234ee587f3f05874723c496099ce9f525dd966515a7fe',
    'unified.metrics.json': 'b5b47e2dc23f85dc7b5427118f5adb0da4f0c9e1e662ead7ae7094597629a268',
    'unified_checkpoint.pt': 'ed1c125b6ec2cadacdabac4091850958d5096008771fc76a3bb9ddc33c49774e',
}


def _validate_archive(archive_path: Path, expected_hashes: dict[str, str]) -> dict[str, bytes]:
    if not archive_path.is_file():
        raise FileNotFoundError(f'Unified export not found: {archive_path}')
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if set(names) != set(expected_hashes) or len(names) != len(set(names)):
            raise ValueError('Unified ZIP must contain exactly the four expected artifact components')
        if any(entry.is_dir() or entry.flag_bits & 1 or entry.filename != Path(entry.filename).name
               or entry.file_size > MAX_COMPONENT_BYTES for entry in entries):
            raise ValueError('Unified ZIP contains an unsafe filename, encrypted entry, directory, or oversized component')
        if archive.testzip() is not None:
            raise ValueError('Unified ZIP CRC check failed')
        content = {name: archive.read(name) for name in expected_hashes}
    for name, value in content.items():
        actual = hashlib.sha256(value).hexdigest()
        if actual != expected_hashes[name].lower():
            raise ValueError(f'Unexpected SHA-256 for {name}; installation refused')
    try:
        report = json.loads(content['unified.metrics.json'])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('Unified metrics manifest is not valid JSON') from exc
    if (report.get('format_version') != 1 or report.get('model_kind') != 'izqar_unified_mtl'
            or report.get('serialization') != 'numpy_shared_backbone_v1'):
        raise ValueError('Unified metrics manifest has unsupported model metadata')
    return content


def install(archive_path: Path, destination: Path, expected_hashes: dict[str, str] | None = None) -> list[Path]:
    """Verify and atomically install a reviewed Unified export without overwrite."""
    expected = TRUSTED_COMPONENT_SHA256 if expected_hashes is None else expected_hashes
    content = _validate_archive(archive_path, expected)
    destination.mkdir(parents=True, exist_ok=True)
    targets = [destination / name for name in expected]
    if any(target.exists() for target in targets):
        raise FileExistsError('Unified artifact already exists; preserve it or choose another destination')
    created: list[Path] = []
    try:
        for target in targets:
            with target.open('xb') as output:
                output.write(content[target.name])
            created.append(target)
    except BaseException:
        for target in created:
            target.unlink(missing_ok=True)
        raise
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='ZIP downloaded from the reviewed Kaggle run')
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1] / 'models')
    args = parser.parse_args()
    for target in install(args.archive, args.destination):
        print(f'Installed: {target}')


if __name__ == '__main__':
    main()
