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


def test_specialist_notebooks_pin_reviewed_code_and_default_to_no_final_test():
    for name in ('ieee_cis_kaggle.ipynb', 'baf_kaggle.ipynb', 'ibm_aml_kaggle.ipynb'):
        parsed = json.loads((ROOT / 'notebooks' / name).read_text(encoding='utf-8'))
        source = '\n'.join(''.join(cell.get('source', [])) for cell in parsed['cells'])
        assert "EXPECTED_REVISION = '" in source
        assert "['git', 'checkout', '--detach', EXPECTED_REVISION]" in source
        assert 'RUN_FINAL = False' in source


def test_ibm_graph_notebook_is_bounded_and_revision_pinned():
    parsed = json.loads((ROOT / 'notebooks' / 'ibm_graph_kaggle.ipynb').read_text(encoding='utf-8'))
    source = '\n'.join(''.join(cell.get('source', [])) for cell in parsed['cells'])
    assert parsed['nbformat'] == 4
    assert "EXPECTED_REVISION = '" in source
    assert "['git', 'checkout', '--detach', EXPECTED_REVISION]" in source
    assert "'analyze_ibm_graph.py'" in source
    assert "'--max-rows', '500000'" in source
    assert "'label_used': False" in source
    assert 'train_ibm_aml.py' not in source
