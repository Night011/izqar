"""Offline-only IBM AML causal-history candidate exports.

No API serving, no final-test evaluation, no automatic model promotion.
Exporting requires a fitted pipeline from a trusted *experiment* run.
Importing verifies metadata and the caller-pinned archive SHA256 before writing.
Optional local smoke deserializes the trusted preprocessor only after verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

BASE_NUMERIC = (
    'amount_paid', 'log_amount_paid', 'same_bank', 'same_currency', 'hour', 'weekday',
)
HISTORY_NUMERIC = (
    'sender_prior_outgoing_count', 'recipient_prior_incoming_count',
    'sender_prior_outgoing_amount', 'recipient_prior_incoming_amount',
    'seconds_since_sender_outgoing', 'seconds_since_recipient_incoming',
)
CATEGORICAL = ('receiving_currency', 'payment_currency', 'payment_format')
CAUSAL_FEATURE_SETS = frozenset({'causal_history_v1', 'causal_history_bank_v2'})
EXPECTED_FEATURES = BASE_NUMERIC + HISTORY_NUMERIC + CATEGORICAL
MAX_BUNDLE_BYTES = 200 * 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_COMPONENT_BYTES = 180 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED = 200 * 1024 * 1024
SHA256_PATTERN = re.compile(r'^[0-9a-fA-F]{64}$')
NAME_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def check_candidate_report(report: dict, *, source: bool = False) -> None:
    if not isinstance(report, dict):
        raise ValueError('IBM shadow report must be a JSON object')
    if report.get('model_kind') != 'ibm_aml' or report.get('run_mode') != 'experiment':
        raise ValueError('Shadow export requires an IBM AML validation-only experiment report')
    experiment = report.get('experiment')
    if not isinstance(experiment, dict) or experiment.get('mode') != 'experiment':
        raise ValueError('Missing validation-only experiment provenance')
    if report.get('test_metrics') is not None or report.get('baseline_test_metrics') is not None:
        raise ValueError('Shadow export must not contain final-test metrics')
    if not isinstance(report.get('features'), list) or tuple(report['features']) != EXPECTED_FEATURES:
        raise ValueError('IBM shadow requires exactly the reviewed causal_history_v1 feature schema')
    if (not isinstance(report.get('numerical_features'), list)
            or tuple(report['numerical_features']) != BASE_NUMERIC + HISTORY_NUMERIC):
        raise ValueError('IBM shadow numeric schema is incompatible')
    if not isinstance(report.get('categorical_features'), list) or tuple(report['categorical_features']) != CATEGORICAL:
        raise ValueError('IBM shadow categorical schema is incompatible')
    if not isinstance(report.get('model_version'), str) or not report['model_version'].strip():
        raise ValueError('IBM shadow model version is missing')
    if (isinstance(report['experiment'].get('hyperparameters'), dict)
            and report['experiment']['hyperparameters'].get('feature_set') == 'causal_history_bank_v2'
            and report['model_version'] == 'ibm-aml-hi-small-xgboost-v1'):
        raise ValueError('Bank-scoped causal-history v2 must have a distinct model version')
    revision = experiment.get('code_revision')
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('IBM shadow requires a pinned 40-character code revision')
    hyperparameters = experiment.get('hyperparameters')
    if not isinstance(hyperparameters, dict) or hyperparameters.get('feature_set') not in CAUSAL_FEATURE_SETS:
        raise ValueError('IBM shadow provenance must identify causal_history_v1 or causal_history_bank_v2')
    data_files = report.get('data_files')
    if (not isinstance(data_files, list) or len(data_files) != 1 or not isinstance(data_files[0], dict)
            or not isinstance(data_files[0].get('sha256'), str)
            or not SHA256_PATTERN.fullmatch(data_files[0]['sha256'])):
        raise ValueError('IBM shadow requires one SHA-256-pinned source dataset')
    libraries = report.get('library_versions')
    if (not isinstance(libraries, dict) or not isinstance(libraries.get('scikit_learn'), str)
            or not isinstance(libraries.get('xgboost'), str)):
        raise ValueError('IBM shadow requires pinned scikit-learn and XGBoost training versions')
    threshold = report.get('threshold')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('IBM shadow threshold must be a finite value between 0 and 1')
    validation = report.get('validation_metrics')
    if not isinstance(validation, dict):
        raise ValueError('IBM shadow must contain validation metrics')
    pr_auc = validation.get('average_precision_pr_auc')
    if (isinstance(pr_auc, bool) or not isinstance(pr_auc, (float, int))
            or not math.isfinite(pr_auc) or not 0 <= pr_auc <= 1):
        raise ValueError('IBM shadow must contain a finite validation PR-AUC')
    if source:
        if report.get('format_version') != 2:
            raise ValueError('Expected a format-v2 training report before portable export')
    elif (report.get('format_version') != 3 or report.get('serialization') != 'portable_xgboost_v1'
          or report.get('deployment_role') != 'offline_shadow_only'
          or report.get('serving_allowed') is not False):
        raise ValueError('Unsafe or unsupported portable IBM shadow manifest')


def _names(stem: str) -> dict[str, str]:
    if not NAME_PATTERN.fullmatch(stem):
        raise ValueError('Shadow bundle name must be a simple ASCII basename')
    return {
        'preprocessor': stem + '.preprocessor.joblib',
        'xgboost': stem + '.xgboost.json',
        'report': stem + '.metrics.json',
        'manifest': stem + '.shadow.json',
    }


def _synthetic_frame():
    """Schema smoke only. These values are NOT produced from a real account history."""
    import pandas as pd
    amount = 125.0
    row = {
        'amount_paid': amount, 'log_amount_paid': math.log1p(amount),
        'same_bank': 0.0, 'same_currency': 1.0, 'hour': 12.0, 'weekday': 2.0,
        'sender_prior_outgoing_count': 2.0, 'recipient_prior_incoming_count': 1.0,
        'sender_prior_outgoing_amount': 250.0, 'recipient_prior_incoming_amount': 100.0,
        'seconds_since_sender_outgoing': 3600.0, 'seconds_since_recipient_incoming': 7200.0,
        'receiving_currency': 'USD', 'payment_currency': 'USD', 'payment_format': 'Wire',
    }
    return pd.DataFrame([row], columns=list(EXPECTED_FEATURES))


def export_shadow_bundle(pipeline, report: dict, destination_zip: Path) -> dict:
    """Export an experiment's trained weights without evaluating or reading test.

    The smoke compares in-memory and reloaded predictions using a synthetic
    contract row. It validates serialization, not model accuracy or causality.
    """
    check_candidate_report(report, source=True)
    if destination_zip.suffix != '.zip':
        raise ValueError('The shadow export destination must end with .zip')
    if destination_zip.exists():
        raise FileExistsError('The shadow ZIP already exists; do not overwrite candidates')
    names = _names(destination_zip.stem)
    if tuple(pipeline.feature_names_in_) != EXPECTED_FEATURES:
        raise ValueError('Fitted pipeline feature names differ from reviewed causal schema')
    from sklearn.pipeline import Pipeline
    if not isinstance(pipeline, Pipeline) or set(pipeline.named_steps) != {'preprocessing', 'model'}:
        raise ValueError('Expected a fitted preprocessing + XGBoost pipeline')
    import joblib
    import numpy as np
    from xgboost import XGBClassifier
    if not isinstance(pipeline.named_steps['model'], XGBClassifier):
        raise ValueError('Shadow model must be an XGBClassifier')
    # Perform a purely synthetic contract-level prediction, not another
    # evaluation or a look at holdout labels.
    row = _synthetic_frame()
    original = float(pipeline.predict_proba(row)[0, 1])
    destination_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='izqar-shadow-export-') as temp:
        directory = Path(temp)
        pre_path = directory / names['preprocessor']
        model_path = directory / names['xgboost']
        joblib.dump(pipeline.named_steps['preprocessing'], pre_path, compress=0)
        pipeline.named_steps['model'].save_model(model_path)
        # Load only objects created in this same trusted training process.
        pre_roundtrip = joblib.load(pre_path)
        model_roundtrip = XGBClassifier()
        model_roundtrip.load_model(model_path)
        roundtrip = float(model_roundtrip.predict_proba(pre_roundtrip.transform(row))[0, 1])
        delta = abs(original - roundtrip)
        if not (math.isfinite(original) and math.isfinite(roundtrip) and 0 <= roundtrip <= 1 and delta <= 1e-6):
            raise ValueError('The IBM shadow portable roundtrip changed the prediction')
        portable = dict(report)
        portable.update({'format_version': 3, 'serialization': 'portable_xgboost_v1',
                         'deployment_role': 'offline_shadow_only', 'serving_allowed': False,
                         'shadow_contract': report['experiment']['hyperparameters']['feature_set'] + ': prior history only; bank/account keys are not ML inputs',
                         'shadow_smoke': {'kind': 'synthetic_schema_only', 'rows': 1,
                                          'max_absolute_score_difference': delta,
                                          'warning': 'A synthetic serialization smoke is not a performance evaluation.'}})
        report_bytes = json_bytes(portable)
        if len(report_bytes) > MAX_METADATA_BYTES:
            raise ValueError('Shadow metadata exceeds size limit')
        (directory / names['report']).write_bytes(report_bytes)
        records = {}
        for key in ('preprocessor', 'xgboost', 'report'):
            source = directory / names[key]
            if source.stat().st_size > (MAX_METADATA_BYTES if key == 'report' else MAX_COMPONENT_BYTES):
                raise ValueError(f'{key} component exceeds limit')
            records[key] = {'name': names[key], 'bytes': source.stat().st_size, 'sha256': sha256(source)}
        manifest = {
            'format_version': 1, 'bundle_type': 'izqar_ibm_causal_offline_shadow',
            'mode': 'experiment', 'serving_allowed': False,
            'feature_set': report['experiment']['hyperparameters']['feature_set'],
            'code_revision': report['experiment']['code_revision'],
            'source_data_sha256': report['data_files'][0]['sha256'].lower(),
            'provenance_warning': 'SHA256 identifies bytes but does not authenticate the model creator.',
            'files': records,
        }
        manifest_bytes = json_bytes(manifest)
        (directory / names['manifest']).write_bytes(manifest_bytes)
        try:
            # Exclusive create protects existing reports; clean an incomplete ZIP.
            with destination_zip.open('xb') as target:
                with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                    for key in ('preprocessor', 'xgboost', 'report', 'manifest'):
                        archive.write(directory / names[key], arcname=names[key])
            verified = verify_shadow_bundle(destination_zip)
        except BaseException:
            destination_zip.unlink(missing_ok=True)
            raise
    return {'archive_path': str(destination_zip), 'sha256': verified['sha256'],
            'synthetic_smoke_max_delta': delta, 'note': 'Offline shadow only; not an API model'}


def verify_shadow_bundle(bundle: Path, expected_sha256: str | None = None) -> dict:
    """Validate archive without ever deserializing the model or preprocessor."""
    if not bundle.is_file() or bundle.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError('Shadow bundle is missing or exceeds the archive limit')
    actual_sha = sha256(bundle)
    if expected_sha256 is not None:
        if not SHA256_PATTERN.fullmatch(expected_sha256) or actual_sha != expected_sha256.lower():
            raise ValueError('Shadow bundle SHA-256 does not match the explicitly pinned value')
    with zipfile.ZipFile(bundle) as archive:
        entries = archive.infolist()
        files = [e.filename for e in entries]
        if len(files) != 4 or len(set(files)) != 4:
            raise ValueError('Shadow bundle must contain exactly four distinct flat files')
        if any(e.is_dir() or (e.flag_bits & 1) or (stat.S_IFMT(e.external_attr >> 16) == stat.S_IFLNK)
               or e.filename != Path(e.filename).name or '/' in e.filename or '\\' in e.filename
               or e.filename.startswith('.') for e in entries):
            raise ValueError('Unsafe shadow archive entry')
        if sum(e.file_size for e in entries) > MAX_TOTAL_UNCOMPRESSED:
            raise ValueError('Excessive uncompressed archive size')
        if any(e.file_size > (MAX_METADATA_BYTES if e.filename.endswith(('.metrics.json', '.shadow.json'))
                              else MAX_COMPONENT_BYTES) for e in entries):
            raise ValueError('Oversized shadow archive component')
        manifest_names = [name for name in files if name.endswith('.shadow.json')]
        if len(manifest_names) != 1:
            raise ValueError('Expected exactly one shadow manifest')
        stem = manifest_names[0][:-len('.shadow.json')]
        names = _names(stem)
        if set(files) != set(names.values()):
            raise ValueError('Shadow archive has missing or unexpected entries')
        manifest = json.loads(archive.read(names['manifest']))
        if (not isinstance(manifest, dict) or manifest.get('format_version') != 1
                or manifest.get('bundle_type') != 'izqar_ibm_causal_offline_shadow'
                or manifest.get('mode') != 'experiment' or manifest.get('serving_allowed') is not False
                or manifest.get('feature_set') not in CAUSAL_FEATURE_SETS):
            raise ValueError('Invalid shadow bundle manifest')
        records = manifest.get('files')
        if not isinstance(records, dict) or set(records) != {'preprocessor', 'xgboost', 'report'}:
            raise ValueError('Invalid shadow bundle contents manifest')
        for role in ('preprocessor', 'xgboost', 'report'):
            entry = archive.getinfo(names[role])
            record = records[role]
            if (not isinstance(record, dict) or record.get('name') != names[role]
                    or record.get('bytes') != entry.file_size
                    or not isinstance(record.get('sha256'), str)
                    or not SHA256_PATTERN.fullmatch(record['sha256'])):
                raise ValueError(f'Invalid metadata for {role} component')
            digest = hashlib.sha256()
            with archive.open(entry) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != record['sha256']:
                raise ValueError(f'SHA-256 mismatch for {role} component')
        if archive.testzip() is not None:
            raise ValueError('Shadow archive CRC failure')
        report = json.loads(archive.read(names['report']))
        check_candidate_report(report)
        if (manifest.get('code_revision') != report['experiment']['code_revision']
                or manifest.get('source_data_sha256') != report['data_files'][0]['sha256'].lower()
                or manifest.get('feature_set') != report['experiment']['hyperparameters']['feature_set']):
            raise ValueError('Shadow bundle provenance differs from the report')
        return {'sha256': actual_sha, 'stem': stem, 'files': names,
                'validation_metrics': report['validation_metrics'],
                'warning': 'Validation-only synthetic research; no bank performance or production serving claim.'}


def install_shadow_bundle(bundle: Path, destination: Path, expected_sha256: str) -> dict:
    """Install candidate files only into an explicitly named experiments folder."""
    verified = verify_shadow_bundle(bundle, expected_sha256=expected_sha256)
    normalized = destination.resolve()
    if 'experiments' not in normalized.parts:
        raise ValueError('Shadow bundles may only be installed under a directory named experiments')
    if destination.is_symlink():
        raise ValueError('Shadow destination must not be a symlink')
    names = verified['files']
    paths = [destination / names[role] for role in ('preprocessor', 'xgboost', 'report', 'manifest')]
    if any(target.exists() or target.is_symlink() for target in paths):
        raise FileExistsError('Existing shadow artifact would be overwritten')
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        with zipfile.ZipFile(bundle) as archive:
            for role, target in zip(('preprocessor', 'xgboost', 'report', 'manifest'), paths):
                with archive.open(names[role]) as source, target.open('xb') as sink:
                    written.append(target)
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
        # Re-check raw extracted files against the manifest after write.
        manifest = json.loads((destination / names['manifest']).read_text(encoding='utf-8'))
        for role in ('preprocessor', 'xgboost', 'report'):
            if sha256(destination / names[role]) != manifest['files'][role]['sha256']:
                raise ValueError('Extracted component verification failed')
    except BaseException:
        for target in written:
            target.unlink(missing_ok=True)
        raise
    return {**verified, 'installed_files': [str(p) for p in paths]}


def verify_installed_shadow(destination: Path, stem: str) -> dict:
    """Rehash every installed component before an explicit trusted pickle smoke.

    This check is intentionally metadata-only. It never calls joblib.load or
    XGBoost model.load_model. The caller must separately authenticate the
    source archive SHA256, because a locally edited manifest cannot certify
    who trained a model.
    """
    names = _names(stem)
    if destination.is_symlink() or 'experiments' not in destination.resolve().parts:
        raise ValueError('Installed shadow must remain in a non-symlink experiments directory')
    for component in names.values():
        path = destination / component
        if path.is_symlink() or not path.is_file():
            raise ValueError('Installed shadow component is missing or a symlink')
        limit = MAX_METADATA_BYTES if component.endswith(('.metrics.json', '.shadow.json')) else MAX_COMPONENT_BYTES
        if path.stat().st_size > limit:
            raise ValueError('Installed shadow component exceeds its declared size limit')
    manifest = json.loads((destination / names['manifest']).read_text(encoding='utf-8'))
    report = json.loads((destination / names['report']).read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError('Installed shadow manifest is not a JSON object')
    check_candidate_report(report)
    if (manifest.get('format_version') != 1 or manifest.get('bundle_type') != 'izqar_ibm_causal_offline_shadow'
            or manifest.get('serving_allowed') is not False or manifest.get('mode') != 'experiment'
            or manifest.get('feature_set') not in CAUSAL_FEATURE_SETS
            or manifest.get('code_revision') != report['experiment']['code_revision']
            or manifest.get('source_data_sha256') != report['data_files'][0]['sha256'].lower()
            or manifest.get('feature_set') != report['experiment']['hyperparameters']['feature_set']):
        raise ValueError('Installed shadow manifest disagrees with its validation-only report')
    records = manifest.get('files')
    if not isinstance(records, dict) or set(records) != {'preprocessor', 'xgboost', 'report'}:
        raise ValueError('Installed shadow component inventory is invalid')
    for role, key in (('preprocessor', 'preprocessor'), ('xgboost', 'xgboost'), ('report', 'report')):
        path = destination / names[key]
        entry = records[role]
        if (not isinstance(entry, dict) or entry.get('name') != names[key]
                or entry.get('bytes') != path.stat().st_size
                or not isinstance(entry.get('sha256'), str)
                or not SHA256_PATTERN.fullmatch(entry['sha256'])
                or sha256(path) != entry['sha256'].lower()):
            raise ValueError(f'Installed shadow {role} component hash or metadata mismatch')
    return {'status': 'verified_local_components', 'stem': stem,
            'code_revision': report['experiment']['code_revision'],
            'warning': 'Local manifest hashes do not independently authenticate the creator of this artifact.'}


def trusted_smoke_installed(destination: Path, stem: str) -> dict:
    """Explicit local smoke; ONLY use for a user-verified, trusted SHA256 ZIP.

    The joblib loader executes Python pickle data. Do not call for untrusted
    exports or expose through the API.
    """
    import joblib
    import numpy as np
    import sklearn
    import xgboost
    from xgboost import XGBClassifier
    # Hash and provenance checks run before opening any pickle payload.
    integrity = verify_installed_shadow(destination, stem)
    names = _names(stem)
    report = json.loads((destination / names['report']).read_text(encoding='utf-8'))
    required_sklearn = report['library_versions']['scikit_learn']
    if required_sklearn != sklearn.__version__:
        raise RuntimeError(f'Shadow needs scikit-learn {required_sklearn}; local runtime has {sklearn.__version__}')
    required_xgboost = report['library_versions']['xgboost']
    if required_xgboost != xgboost.__version__:
        raise RuntimeError(f'Shadow needs XGBoost {required_xgboost}; local runtime has {xgboost.__version__}')
    preprocessor = joblib.load(destination / names['preprocessor'])
    model = XGBClassifier()
    model.load_model(destination / names['xgboost'])
    scores = model.predict_proba(preprocessor.transform(_synthetic_frame()))
    score = float(np.asarray(scores)[0, 1])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Trusted shadow smoke produced an invalid score')
    return {'kind': 'synthetic_schema_only', 'model_load': 'passed', 'predict': 'passed',
            'integrity': integrity['status'], 'uncalibrated_score': score,
            'warning': 'Not real validation performance and not a causal history replay.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    verify = subs.add_parser('verify', help='Inspect the report-only ZIP without unpickling')
    verify.add_argument('--bundle', type=Path, required=True)
    verify.add_argument('--expected-sha256')
    install = subs.add_parser('install', help='Install ONLY under models/experiments')
    install.add_argument('--bundle', type=Path, required=True)
    install.add_argument('--expected-sha256', required=True)
    install.add_argument('--destination', type=Path, required=True)
    install.add_argument('--trusted-smoke', action='store_true',
                         help='Explicitly deserialize a SHA-pinned trusted preprocessor for synthetic smoke')
    local = subs.add_parser('verify-installed', help='Rehash offline installed components; never unpickle')
    local.add_argument('--destination', type=Path, required=True)
    local.add_argument('--stem', required=True)
    args = parser.parse_args()
    if args.command == 'verify-installed':
        result = verify_installed_shadow(args.destination, args.stem)
    elif args.command == 'verify':
        result = verify_shadow_bundle(args.bundle, args.expected_sha256)
    else:
        result = install_shadow_bundle(args.bundle, args.destination, args.expected_sha256)
        if args.trusted_smoke:
            result['trusted_synthetic_smoke'] = trusted_smoke_installed(args.destination, result['stem'])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
