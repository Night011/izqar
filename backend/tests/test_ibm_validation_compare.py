"""Validation comparison guardrails for IBM candidate review, no data files needed."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_ibm_validation import compare_reports, load_report  # noqa: E402


def fixture_report(*, candidate=False):
    return {
        'model_kind': 'ibm_aml', 'run_mode': 'experiment' if candidate else 'final',
        'experiment': {'mode': 'experiment' if candidate else 'final',
                       'hyperparameters': {'max_false_alerts_per_10000': 100}},
        'test_metrics': None if candidate else {'not_read_by_this_tool': True},
        'baseline_test_metrics': None,
        'data_files': [{'filename': 'HI-Small_Trans.csv', 'sha256': 'a' * 64}],
        'split_boundaries': {'validation_start': '2025-02-01', 'test_start': '2025-03-01'},
        'split_counts': {'train': 700, 'validation': 100, 'test': 100},
        'validation_metrics': {'cases': 100, 'average_precision_pr_auc': .04 if not candidate else .12,
                               'roc_auc': .9 if not candidate else .95,
                               'precision': .04 if not candidate else .07,
                               'recall': .35 if not candidate else .65,
                               'f1': .07 if not candidate else .12,
                               'false_alerts_per_10000_operations': 90 if not candidate else 87},
    }


def test_ibm_comparison_reports_same_window_improvement():
    result = compare_reports(fixture_report(), fixture_report(candidate=True))
    assert result['observations']['improved_all_three'] is True
    assert result['delta_candidate_minus_baseline']['recall'] == pytest.approx(.3)
    assert result['decision'] == 'research_only_no_automatic_promotion'
    assert result['baseline']['cases'] == 100


@pytest.mark.parametrize('alter,match', [
    (lambda r: r['data_files'][0].update(sha256='b' * 64), 'Different source'),
    (lambda r: r['split_boundaries'].update(test_start='2025-04-01'), 'Different chronological'),
    (lambda r: r['validation_metrics'].update(cases=50), 'population sizes'),
    (lambda r: r['experiment']['hyperparameters'].update(max_false_alerts_per_10000=150), 'Different false-alert'),
    (lambda r: r.update(test_metrics={'f1': .95}), 'validation-only'),
    (lambda r: r['validation_metrics'].update(f1=float('nan')), 'Invalid validation metric'),
])
def test_comparator_rejects_unfair_or_unsafe_comparison(alter, match):
    candidate = fixture_report(candidate=True)
    alter(candidate)
    with pytest.raises(ValueError, match=match):
        compare_reports(fixture_report(), candidate)


def test_comparator_requires_explicit_source_provenance():
    candidate = fixture_report(candidate=True)
    del candidate['data_files'][0]['sha256']
    with pytest.raises(ValueError, match='Invalid source'):
        compare_reports(fixture_report(), candidate)


def test_comparison_json_reader_rejects_nonfinite_and_oversized(tmp_path):
    path = tmp_path / 'candidate.json'
    path.write_text('{"model_kind":"ibm_aml", "a":NaN}')
    with pytest.raises(ValueError, match='Non-finite'):
        load_report(path)
    path.write_text(json.dumps(fixture_report(candidate=True)))
    assert load_report(path)['model_kind'] == 'ibm_aml'
    path.write_bytes(b'x' * (2 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match='oversized'):
        load_report(path)
