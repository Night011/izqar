"""Install an artifact ZIP exported by a trusted IZQAR training run.

The ZIP is inspected without unpickling. Installing a bundle is a trust
decision: the API will execute its joblib payload when the model is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

import sklearn

MAX_MODEL_BYTES = 200 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
FORBIDDEN_FEATURES = {
    'isFraud', 'isFlaggedFraud', 'newbalanceOrig', 'newbalanceDest',
    'TransactionID', 'TransactionDT',
}


def install(bundle: Path, destination: Path, expected_sha256: str | None = None) -> list[Path]:
    if expected_sha256 is not None:
        actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise ValueError('Bundle SHA-256 does not match the expected digest')
    with zipfile.ZipFile(bundle) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        reports = [name for name in names if name.endswith('.metrics.json')]
        if len(reports) != 1 or len(names) != len(set(names)):
            raise ValueError('Bundle must contain exactly one report and unique filenames')
        report_name = reports[0]
        stem = report_name[:-len('.metrics.json')]
        audit_name = stem + '.validation_audit.json'
        if any(name != Path(name).name or '/' in name or '\\' in name or name.startswith('.') for name in names):
            raise ValueError('Bundle filenames must be plain basenames')
        if any(entry.is_dir() or entry.flag_bits & 1 for entry in entries):
            raise ValueError('Directories and encrypted entries are unsupported')
        if any(entry.file_size > (MAX_JSON_BYTES if entry.filename in {report_name, audit_name} else MAX_MODEL_BYTES) for entry in entries):
            raise ValueError('Bundle entry exceeds the size limit')
        report = json.loads(archive.read(report_name))
        portable = report.get('format_version') == 3 and report.get('serialization') == 'portable_xgboost_v1'
        model_files = ({stem + '.preprocessor.joblib', stem + '.xgboost.json'} if portable else {stem + '.joblib'})
        allowed = model_files | {report_name, audit_name}
        if not model_files <= set(names) or set(names) - allowed:
            raise ValueError('Bundle contains unsupported or missing files')
        if archive.testzip() is not None:
            raise ValueError('Bundle CRC check failed')
        if report.get('model_kind') not in {'paysim', 'ieee_cis', 'baf_base'} or report.get('model_version') is None:
            raise ValueError('Missing or unsupported model metadata')
        if report.get('library_versions', {}).get('scikit_learn') != sklearn.__version__:
            raise ValueError(f"Model requires scikit-learn {report.get('library_versions', {}).get('scikit_learn')}; runtime has {sklearn.__version__}")
        features = report.get('features')
        if not isinstance(features, list) or not features or any(not isinstance(feature, str) for feature in features):
            raise ValueError('Missing model feature schema')
        if set(features) & FORBIDDEN_FEATURES or len(features) != len(set(features)):
            raise ValueError('Model feature schema contains a forbidden or duplicate feature')
        threshold = report.get('threshold')
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Invalid model threshold')
        if report.get('test_metrics') is None:
            raise ValueError('Bundle has no recorded test metrics')
        if audit_name in names:
            audit = json.loads(archive.read(audit_name))
            if audit.get('model_version') != report['model_version'] or 'test_metrics' in audit:
                raise ValueError('Validation audit does not match the model')
        destination.mkdir(parents=True, exist_ok=True)
        targets = [destination / name for name in names]
        if any(target.exists() for target in targets):
            raise FileExistsError('An artifact already exists; choose a new filename or preserve the installed version')
        created: list[Path] = []
        try:
            for name, target in zip(names, targets):
                with archive.open(name) as source, target.open('xb') as output:
                    created.append(target)
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        except BaseException:
            for target in created:
                target.unlink(missing_ok=True)
            raise
        return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    for path in install(args.bundle, args.destination, args.expected_sha256):
        print(f'Installed: {path}')


if __name__ == '__main__':
    main()
