"""Read-only comparison of two IBM AML reports on the SAME validation window.

Never deserializes weights, never loads test data, never alters a threshold,
never installs or promotes a model. An increase in one metric alone is not
proof of production readiness.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MAX_REPORT_BYTES = 2 * 1024 * 1024
METRICS = (
    'average_precision_pr_auc', 'roc_auc', 'precision', 'recall', 'f1',
    'false_alerts_per_10000_operations',
)


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON value {value} is forbidden')


def load_report(path: Path) -> dict:
    """Accept a local report JSON only. No joblib, ZIP weights or model code."""
    if not path.is_file() or path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError('Missing or oversized validation report')
    report = json.loads(path.read_text(encoding='utf-8'), parse_constant=_reject_constant)
    if not isinstance(report, dict) or report.get('model_kind') != 'ibm_aml':
        raise ValueError('Expected an IBM AML report')
    return report


def _sha(report: dict) -> str:
    files = report.get('data_files')
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise ValueError('Missing source dataset provenance')
    digest = files[0].get('sha256')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdefABCDEF' for c in digest):
        raise ValueError('Invalid source dataset SHA-256')
    return digest.lower()


def _metrics(report: dict) -> dict:
    obj = report.get('validation_metrics')
    if not isinstance(obj, dict):
        raise ValueError('Missing validation metrics')
    out = {}
    for key in METRICS:
        value = obj.get(key)
        if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
                or value < 0 or (value > 10000 if key == 'false_alerts_per_10000_operations' else value > 1)):
            raise ValueError(f'Invalid validation metric {key}')
        out[key] = float(value)
    count = obj.get('cases')
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Validation cases must be a positive integer')
    out['cases'] = count
    return out


def _budget(report: dict):
    experiment = report.get('experiment')
    hyperparameters = experiment.get('hyperparameters') if isinstance(experiment, dict) else None
    value = hyperparameters.get('max_false_alerts_per_10000') if isinstance(hyperparameters, dict) else None
    if (value is None or isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 10000):
        raise ValueError('Both runs must explicitly declare a comparable FP budget')
    return float(value)


def compare_reports(baseline: dict, candidate: dict) -> dict:
    """Fail on different validation populations, even when metrics seem better."""
    if baseline.get('model_kind') != 'ibm_aml' or candidate.get('model_kind') != 'ibm_aml':
        raise ValueError('Only IBM AML reports can be compared')
    if candidate.get('run_mode') != 'experiment' or candidate.get('test_metrics') is not None:
        raise ValueError('Candidate must be validation-only without final-test metrics')
    if candidate.get('baseline_test_metrics') is not None:
        raise ValueError('Candidate must not contain baseline final-test metrics')
    eb = candidate.get('experiment')
    if not isinstance(eb, dict) or eb.get('mode') != 'experiment':
        raise ValueError('Candidate experiment mode is not confirmed')
    if _sha(baseline) != _sha(candidate):
        raise ValueError('Different source dataset SHA-256, comparison forbidden')
    boundaries = (baseline.get('split_boundaries'), candidate.get('split_boundaries'))
    if not isinstance(boundaries[0], dict) or boundaries[0] != boundaries[1]:
        raise ValueError('Different chronological validation boundaries, comparison forbidden')
    left, right = _metrics(baseline), _metrics(candidate)
    if left['cases'] != right['cases']:
        raise ValueError('Different validation population sizes, comparison forbidden')
    base_counts, new_counts = baseline.get('split_counts'), candidate.get('split_counts')
    if not isinstance(base_counts, dict) or not isinstance(new_counts, dict):
        raise ValueError('Validation split count metadata is missing')
    base_count = base_counts.get('validation')
    new_count = new_counts.get('validation')
    if base_count != left['cases'] or new_count != right['cases']:
        raise ValueError('Validation split count disagrees with the metric report')
    budget0, budget1 = _budget(baseline), _budget(candidate)
    if budget0 != budget1:
        raise ValueError('Different false-alert budgets, comparison forbidden')
    delta = {key: round(right[key] - left[key], 6) for key in METRICS}
    pr_up = delta['average_precision_pr_auc'] > 0
    recall_up = delta['recall'] > 0
    queue_safe = delta['false_alerts_per_10000_operations'] <= 0.001
    return {
        'comparison_kind': 'ibm_same_window_validation_only',
        'same_source_sha256': _sha(candidate),
        'validation_window': candidate['split_boundaries'],
        'validation_cases': right['cases'],
        'max_false_alerts_per_10000': budget1,
        'baseline': left, 'candidate': right, 'delta_candidate_minus_baseline': delta,
        'observations': {
            'pr_auc_increased': pr_up,
            'recall_increased': recall_up,
            'false_alert_queue_not_increased': queue_safe,
            'improved_all_three': pr_up and recall_up and queue_safe,
        },
        'decision': 'research_only_no_automatic_promotion',
        'warning': 'Comparison reuses validation. It is not independent final-test or real-bank evidence.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Optional new JSON report; never overwrites')
    args = parser.parse_args()
    comparison = compare_reports(load_report(args.baseline), load_report(args.candidate))
    payload = json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as sink:
            sink.write(payload)
    print(payload, end='')


if __name__ == '__main__':
    main()
