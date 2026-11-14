"""Safely expose an optional, aggregate IBM AML graph-analysis report.

The offline analyser may contain synthetic account identifiers in its detailed
leads.  The API deliberately returns counts and run limits only: identifiers
and chains stay in the local analyst export rather than being sent to the UI.
"""
from __future__ import annotations

import json
from pathlib import Path


MAX_REPORT_BYTES = 1_048_576
SIGNAL_KEYS = ('fan_in', 'fan_out', 'rapid_transfer_chains', 'directed_three_node_cycles')


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f'Invalid IBM graph report field: {name}')
    return value


def read_public_summary(path: Path) -> dict:
    """Return a bounded, identifier-free view of a trusted local report."""
    if not path.is_file():
        return {'available': False, 'warning': 'IBM AML graph report has not been imported yet.'}
    if path.stat().st_size > MAX_REPORT_BYTES:
        return {'available': False, 'warning': 'IBM AML graph report is too large to expose.'}
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {'available': False, 'warning': 'IBM AML graph report is unreadable.'}
    if not isinstance(report, dict):
        return {'available': False, 'warning': 'IBM AML graph report is invalid.'}
    try:
        rows = _nonnegative_int(report.get('rows_analyzed'), 'rows_analyzed')
        edges = _nonnegative_int(report.get('edge_count'), 'edge_count')
        limits = report.get('limits')
        if not isinstance(limits, dict):
            raise ValueError('limits')
        clean_limits = {key: _nonnegative_int(limits.get(key), f'limits.{key}')
                        for key in ('max_rows', 'rapid_seconds', 'min_degree', 'max_results')}
        counts = {}
        for key in SIGNAL_KEYS:
            items = report.get(key)
            if not isinstance(items, list):
                raise ValueError(key)
            counts[key] = len(items)
        warning = report.get('warning')
        if not isinstance(warning, str) or not warning.strip() or len(warning) > 500:
            raise ValueError('warning')
    except ValueError:
        return {'available': False, 'warning': 'IBM AML graph report failed validation.'}
    return {'available': True, 'source': 'IBM AML HI-Small synthetic transfers',
            'rows_analyzed': rows, 'edge_count': edges, 'limits': clean_limits,
            'signal_counts': counts, 'warning': warning}
