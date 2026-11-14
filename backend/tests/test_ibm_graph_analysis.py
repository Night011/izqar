"""Bounded IBM graph signals must be causal and independent of AML labels."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_ibm_graph import analyze  # noqa: E402
from train_ibm_aml import EXPECTED_HEADER  # noqa: E402


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.write_text(','.join(EXPECTED_HEADER) + '\n' + '\n'.join(','.join(row) for row in rows) + '\n', encoding='utf-8')


def test_ibm_graph_finds_bounded_fan_chain_and_cycle_without_labels(tmp_path):
    path = tmp_path / 'HI-Small_Trans.csv'
    write_csv(path, [
        ['2022-01-01 00:00:00', '1', 'A', '2', 'B', '1', 'USD', '10', 'USD', 'ACH', '0'],
        ['2022-01-01 00:01:00', '1', 'B', '2', 'C', '1', 'USD', '9', 'USD', 'ACH', '1'],
        ['2022-01-01 00:02:00', '1', 'C', '2', 'A', '1', 'USD', '8', 'USD', 'ACH', '0'],
        ['2022-01-01 00:03:00', '3', 'D', '2', 'B', '1', 'USD', '7', 'USD', 'ACH', '1'],
        ['2022-01-01 00:04:00', '4', 'E', '2', 'B', '1', 'USD', '6', 'USD', 'ACH', '0'],
    ])
    report = analyze(path, max_rows=100, rapid_seconds=120, min_degree=3)
    assert report['rows_analyzed'] == 5
    assert report['fan_in'] == [{'account': 'B', 'incoming_transfers': 3, 'incoming_amount': 23.0}]
    assert report['directed_three_node_cycles'] == [['A', 'B', 'C', 'A']]
    assert report['rapid_transfer_chains'][0]['path'] == ['A', 'B', 'C']


def test_ibm_graph_rejects_non_chronological_input(tmp_path):
    path = tmp_path / 'HI-Small_Trans.csv'
    write_csv(path, [
        ['2022-01-01 00:01:00', '1', 'A', '2', 'B', '1', 'USD', '10', 'USD', 'ACH', '0'],
        ['2022-01-01 00:00:00', '1', 'B', '2', 'C', '1', 'USD', '9', 'USD', 'ACH', '1'],
    ])
    with pytest.raises(ValueError, match='chronologically ordered'):
        analyze(path)


def test_ibm_graph_accepts_hi_small_slash_minute_timestamps(tmp_path):
    path = tmp_path / 'HI-Small_Trans.csv'
    write_csv(path, [
        ['2022/09/01 00:20', '1', 'A', '2', 'B', '1', 'USD', '10', 'USD', 'ACH', '0'],
        ['2022/09/01 00:21', '1', 'B', '2', 'C', '1', 'USD', '9', 'USD', 'ACH', '1'],
    ])
    report = analyze(path, max_rows=2, rapid_seconds=120)
    assert report['rows_analyzed'] == 2
    assert report['rapid_transfer_chains'][0]['path'] == ['A', 'B', 'C']
