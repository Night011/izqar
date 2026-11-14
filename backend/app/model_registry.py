"""Read-only inventory of local ML reports; never deserialize model payloads."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_ENTRIES = 100
KINDS = {'paysim', 'ieee_cis', 'baf_base', 'ibm_aml', 'izqar_unified_mtl'}
FORBIDDEN = {'isFraud', 'isFlaggedFraud', 'newbalanceOrig', 'newbalanceDest',
             'TransactionID', 'TransactionDT', 'fraud_bool', 'Is Laundering',
             'Account', 'Account.1', 'nameOrig', 'nameDest', 'step', 'month',
             'device_fraud_count', 'days_since_request', 'credit_risk_score',
             'Amount Received', 'origin_balance_after_requested_amount'}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _report_path(path: Path) -> Path:
    return path if path.name.endswith('.metrics.json') else path.with_suffix('.metrics.json')


def _components(report_path: Path, report: dict) -> list[Path]:
    stem = report_path.name.removesuffix('.metrics.json')
    if report.get('format_version') == 1 and report.get('serialization') == 'numpy_shared_backbone_v1':
        return [report_path.with_name(stem + '_preprocessors.joblib'), report_path.with_name(stem + '_weights.npz')]
    if report.get('format_version') == 3 and report.get('serialization') == 'portable_xgboost_v1':
        return [report_path.with_name(stem + '.preprocessor.joblib'),
                report_path.with_name(stem + '.xgboost.json')]
    if report.get('format_version') == 2:
        return [report_path.with_name(stem + '.joblib')]
    return []


def _entry(report_path: Path, active_kind: str | None, lifecycle: str, report_root: Path) -> dict:
    """Expose a stable relative identifier even when report filenames collide."""
    try:
        relative_path = report_path.resolve().relative_to(report_root.resolve()).as_posix()
    except ValueError:
        relative_path = report_path.name
    result = {'manifest': report_path.name, 'active_for': active_kind,
              'report_path': relative_path, 'state': 'invalid', 'lifecycle': lifecycle, 'error': None}
    try:
        if not report_path.is_file():
            raise ValueError('metadata missing')
        if report_path.stat().st_size > MAX_REPORT_BYTES:
            raise ValueError('metadata exceeds size limit')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        kind = report.get('model_kind')
        if kind not in KINDS or (active_kind is not None and kind != active_kind):
            raise ValueError('model kind mismatch')
        if kind == 'izqar_unified_mtl':
            tasks = report.get('task_metadata', {})
            if set(tasks) != {'paysim', 'ieee_cis', 'baf_base', 'ibm_aml'}:
                raise ValueError('unified task schema is incomplete')
            features = []
            task_schemas = {}
            for task, metadata in tasks.items():
                schema = metadata.get('feature_schema', {}).get('all')
                if not isinstance(schema, list) or not schema or any(not isinstance(x, str) for x in schema):
                    raise ValueError(f'invalid unified {task} feature schema')
                if len(schema) != len(set(schema)) or set(schema) & FORBIDDEN:
                    raise ValueError(f'forbidden or duplicate unified {task} feature')
                task_schemas[task] = schema
                features.extend(f'{task}.{name}' for name in schema)
            threshold = None
        else:
            task_schemas = None
            features = report.get('features')
            threshold = report.get('threshold')
        if not isinstance(features, list) or not features or any(not isinstance(x, str) for x in features):
            raise ValueError('invalid feature schema')
        if len(set(features)) != len(features) or (kind != 'izqar_unified_mtl' and set(features) & FORBIDDEN):
            raise ValueError('forbidden or duplicate model feature')
        if kind != 'izqar_unified_mtl' and (not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError('invalid threshold')
        files = _components(report_path, report)
        if not files:
            raise ValueError('unsupported artifact format')
        components = [{'filename': path.name, 'bytes': path.stat().st_size, 'sha256': _sha256(path)}
                      for path in files if path.is_file()]
        result.update({'model_kind': kind, 'model_version': report.get('model_version'),
                       'dataset': report.get('dataset'), 'trained_at_utc': report.get('trained_at_utc'),
                       'format_version': report.get('format_version'), 'feature_count': len(features),
                       'features': features, 'threshold': threshold,
                       'task_schemas': task_schemas, 'experimental': kind == 'izqar_unified_mtl',
                       'data_files': report.get('data_files', []),
                       'split_counts': report.get('split_counts', {}),
                       'validation_metrics': report.get('validation_metrics') or report.get('tasks'),
                       'test_metrics': report.get('test_metrics'),
                       'manifest_sha256': _sha256(report_path), 'components': components,
                       # Integrity state describes the files.  Lifecycle deliberately describes
                       # whether this report is serving traffic: a complete candidate is never
                       # promoted simply because its component files happen to be present.
                       'state': 'installed' if len(components) == len(files) else 'report_only'})
        if len(components) != len(files):
            result['error'] = 'model component missing; report alone is not an installed model'
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result['error'] = str(exc)
    return result


def _lifecycle(report_path: Path, report_root: Path, active_kind: str | None) -> str:
    """Classify deployment intent without loading a model payload.

    Reports written by experiment mode go to ``models/experiments`` and contain
    an experiment mode record.  Either marker is enough to keep the entry out of
    the serving set.  A final report outside that directory remains an archived
    report until an operator explicitly configures it as an active model path.
    """
    if active_kind is not None:
        return 'active'
    try:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        mode = report.get('run_mode')
        if not isinstance(mode, str):
            experiment = report.get('experiment')
            mode = experiment.get('mode') if isinstance(experiment, dict) else None
        relative_parts = report_path.resolve().relative_to(report_root.resolve()).parts
        if mode == 'experiment' or 'experiments' in relative_parts:
            return 'candidate'
    except (OSError, ValueError, TypeError):
        # _entry will expose the metadata error.  Treat unknown archive intent
        # conservatively: it must never become a serving model.
        pass
    return 'archived'


def list_registry(active_paths: dict[str, Path], model_root: Path) -> dict:
    """Inventory active and archived reports without trusting their pickle bytes."""
    entries = []
    seen = set()
    for kind, path in active_paths.items():
        report_path = _report_path(path)
        seen.add(report_path.resolve())
        entries.append(_entry(report_path, kind, _lifecycle(report_path, model_root, kind), model_root))
    if model_root.is_dir():
        root = model_root.resolve()
        for report_path in sorted(model_root.rglob('*.metrics.json')):
            resolved = report_path.resolve()
            if resolved in seen or not resolved.is_relative_to(root) or len(entries) >= MAX_ENTRIES:
                continue
            seen.add(resolved)
            entries.append(_entry(report_path, None, _lifecycle(report_path, root, None), root))
    active = [item for item in entries if item['active_for'] is not None]
    history = [item for item in entries if item['active_for'] is None]
    history.sort(key=lambda item: (str(item.get('trained_at_utc') or ''), item['manifest']),
                 reverse=True)
    return {'registry_version': 2, 'entries': active + history,
            'note': 'Metrics are read from trusted local training reports; checksums verify file identity, not model quality. Candidate and archived reports never receive serving traffic until explicitly configured as active.'}
