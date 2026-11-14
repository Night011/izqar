"""Static guards for Kaggle notebooks that must not silently evaluate old tests."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_paysim_recall_candidate_notebook_is_report_only_and_revision_pinned():
    notebook = ROOT / 'notebooks' / 'paysim_recall_candidate_kaggle.ipynb'
    parsed = json.loads(notebook.read_text(encoding='utf-8'))
    source = '\n'.join(''.join(cell.get('source', [])) for cell in parsed['cells'])
    assert parsed['nbformat'] == 4
    assert "EXPECTED_REVISION = '" in source
    assert "['git', 'checkout', '--detach', EXPECTED_REVISION]" in source
    assert "'--mode', 'experiment'" in source
    assert "'--positive-weight-multiplier', '1.25'" in source
    assert "assert not output.exists()" in source
    assert "metadata['test_metrics'] is None" in source
    assert "archive.write(report, report.name)" in source
    assert 'archive.write(output, output.name)' not in source
