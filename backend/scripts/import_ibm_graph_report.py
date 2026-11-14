"""Validate and install a bounded, provenance-pinned IBM graph report."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
from app.ibm_graph_report import MAX_REPORT_BYTES, read_public_summary


MAX_MANIFEST_BYTES = 64 * 1024
REVISION = re.compile(r'^[0-9a-f]{40}$')


def _json(path: Path, limit: int) -> dict:
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f'{path.name} is missing or exceeds its size limit')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'{path.name} is not valid UTF-8 JSON') from exc
    if not isinstance(value, dict):
        raise ValueError(f'{path.name} must contain an object')
    return value


def _contains_label_key(value: object) -> bool:
    if isinstance(value, dict):
        return 'Is Laundering' in value or any(_contains_label_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_label_key(item) for item in value)
    return False


def import_report(report_path: Path, manifest_path: Path, output_path: Path) -> dict:
    """Copy a trusted graph report without exposing details through this script."""
    report = _json(report_path, MAX_REPORT_BYTES)
    manifest = _json(manifest_path, MAX_MANIFEST_BYTES)
    if _contains_label_key(report):
        raise ValueError('Graph report must not contain the AML target label')
    if (not isinstance(manifest.get('expected_revision'), str)
            or not REVISION.fullmatch(manifest['expected_revision'])
            or manifest.get('actual_revision') != manifest['expected_revision']
            or manifest.get('dataset_input_file') != 'HI-Small_Trans.csv'
            or not isinstance(manifest.get('dataset_input_bytes'), int)
            or manifest['dataset_input_bytes'] <= 0
            or manifest.get('label_used') is not False
            or manifest.get('report_contains_synthetic_account_leads') is not True
            or not isinstance(manifest.get('max_rows'), int)
            or manifest['max_rows'] < 1):
        raise ValueError('Graph run manifest does not prove a label-free pinned run')
    summary = read_public_summary(report_path)
    if not summary.get('available'):
        raise ValueError(f"Graph report is not safe to expose: {summary.get('warning')}")
    if (report.get('source') != 'IBM AML HI-Small synthetic transfers'
            or report.get('rows_analyzed', 0) < 1
            or report['rows_analyzed'] > manifest['max_rows']
            or report.get('limits', {}).get('max_rows') != manifest['max_rows']):
        raise ValueError('Graph report and manifest disagree on bounded source scope')
    manifest_output = output_path.with_suffix('.run.json')
    if output_path.exists() or manifest_output.exists():
        raise FileExistsError('Refusing to overwrite an existing IBM graph report')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report_path, output_path)
    shutil.copyfile(manifest_path, manifest_output)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {'report': str(output_path), 'run_manifest': str(manifest_output),
            'sha256': digest, 'public_summary': summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--run-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('models/experiments/ibm_graph_report.json'))
    args = parser.parse_args()
    print(json.dumps(import_report(args.report, args.run_manifest, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
