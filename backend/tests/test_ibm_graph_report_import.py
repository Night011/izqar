"""The graph report importer accepts only bounded, label-free pinned runs."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from import_ibm_graph_report import import_report  # noqa: E402


REVISION = 'a' * 40


def write_report(path: Path) -> None:
    path.write_text(json.dumps({
        'source': 'IBM AML HI-Small synthetic transfers', 'rows_analyzed': 2, 'edge_count': 2,
        'limits': {'max_rows': 2, 'rapid_seconds': 900, 'min_degree': 3, 'max_results': 50},
        'fan_in': [], 'fan_out': [], 'rapid_transfer_chains': [], 'directed_three_node_cycles': [],
        'warning': 'Structural signals are investigation leads, not proof.',
    }), encoding='utf-8')


def write_manifest(path: Path, **overrides: object) -> None:
    value = {'expected_revision': REVISION, 'actual_revision': REVISION,
             'dataset_input_file': 'HI-Small_Trans.csv', 'dataset_input_bytes': 100,
             'max_rows': 2, 'label_used': False, 'report_contains_synthetic_account_leads': True}
    value.update(overrides)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_imports_pinned_label_free_graph_report(tmp_path):
    report, run, output = tmp_path / 'report.json', tmp_path / 'run.json', tmp_path / 'models' / 'ibm_graph_report.json'
    write_report(report); write_manifest(run)
    result = import_report(report, run, output)
    assert output.is_file() and output.with_suffix('.run.json').is_file()
    assert len(result['sha256']) == 64
    assert result['public_summary']['available'] is True


def test_rejects_graph_report_with_target_label_or_invalid_provenance(tmp_path):
    report, run = tmp_path / 'report.json', tmp_path / 'run.json'
    write_report(report); write_manifest(run, label_used=True)
    with pytest.raises(ValueError, match='label-free pinned'):
        import_report(report, run, tmp_path / 'out.json')
    write_manifest(run)
    report.write_text(json.dumps({'Is Laundering': 1}), encoding='utf-8')
    with pytest.raises(ValueError, match='target label'):
        import_report(report, run, tmp_path / 'out.json')
