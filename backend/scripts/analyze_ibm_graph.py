"""Stream a bounded, time-aware graph analysis of IBM AML HI-Small.

This is an investigation aid for synthetic transfers, not a laundering label
predictor.  It deliberately does not read ``Is Laundering``.  The source must
be chronological so every rapid-chain signal is based only on prior rows.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from train_ibm_aml import EXPECTED_HEADER, validate_file


MAX_ROWS_DEFAULT = 500_000


def _timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f'Invalid IBM AML timestamp: {value!r}') from exc


def analyze(path: Path, max_rows: int = MAX_ROWS_DEFAULT, rapid_seconds: int = 900,
            min_degree: int = 3, max_results: int = 50) -> dict:
    """Summarise directed transfer structure without using labels or future history."""
    if max_rows < 1 or rapid_seconds < 1 or min_degree < 2 or max_results < 1:
        raise ValueError('limits must be positive and min_degree must be at least 2')
    validate_file(path)
    in_count, out_count = Counter(), Counter()
    in_amount, out_amount = Counter(), Counter()
    # Keep unique edges only for the bounded sample; never serialise the full graph.
    edges: set[tuple[str, str]] = set()
    latest_inbound: dict[str, tuple[datetime, str, float]] = {}
    rapid_chains: list[dict] = []
    rows = 0
    previous_time: datetime | None = None
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != EXPECTED_HEADER:
            raise ValueError('Unexpected IBM AML header')
        for raw in reader:
            if rows >= max_rows:
                break
            if len(raw) != len(EXPECTED_HEADER):
                raise ValueError('Malformed IBM AML row')
            at = _timestamp(raw[0])
            if previous_time is not None and at < previous_time:
                raise ValueError('IBM graph input must be chronologically ordered')
            previous_time = at
            sender, recipient = raw[2], raw[4]
            if not sender or not recipient:
                raise ValueError('IBM graph analysis requires nonempty account identifiers')
            try:
                amount = float(raw[7])
            except ValueError as exc:
                raise ValueError('Amount Paid must be numeric') from exc
            if amount < 0:
                raise ValueError('Amount Paid must be nonnegative')
            inbound = latest_inbound.get(sender)
            if inbound is not None:
                earlier_at, origin, earlier_amount = inbound
                delta = (at - earlier_at).total_seconds()
                if 0 <= delta <= rapid_seconds and len(rapid_chains) < max_results:
                    rapid_chains.append({'path': [origin, sender, recipient],
                                         'seconds_between': round(delta, 3),
                                         'amounts': [round(earlier_amount, 2), round(amount, 2)]})
            latest_inbound[recipient] = (at, sender, amount)
            out_count[sender] += 1; out_amount[sender] += amount
            in_count[recipient] += 1; in_amount[recipient] += amount
            edges.add((sender, recipient))
            rows += 1
    fan_in = sorted(({'account': account, 'incoming_transfers': count,
                      'incoming_amount': round(in_amount[account], 2)}
                     for account, count in in_count.items() if count >= min_degree),
                    key=lambda item: (-item['incoming_transfers'], item['account']))[:max_results]
    fan_out = sorted(({'account': account, 'outgoing_transfers': count,
                       'outgoing_amount': round(out_amount[account], 2)}
                      for account, count in out_count.items() if count >= min_degree),
                     key=lambda item: (-item['outgoing_transfers'], item['account']))[:max_results]
    adjacency: dict[str, set[str]] = {}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
    cycles, seen = [], set()
    for first in sorted(adjacency):
        for second in sorted(adjacency[first]):
            for third in sorted(adjacency.get(second, ())):
                if first not in adjacency.get(third, ()) or len({first, second, third}) != 3:
                    continue
                rotation = min((first, second, third), (second, third, first), (third, first, second))
                if rotation not in seen and len(cycles) < max_results:
                    seen.add(rotation); cycles.append([*rotation, rotation[0]])
    return {'source': 'IBM AML HI-Small synthetic transfers', 'rows_analyzed': rows,
            'edge_count': len(edges), 'limits': {'max_rows': max_rows, 'rapid_seconds': rapid_seconds,
                                                  'min_degree': min_degree, 'max_results': max_results},
            'fan_in': fan_in, 'fan_out': fan_out, 'rapid_transfer_chains': rapid_chains,
            'directed_three_node_cycles': cycles,
            'warning': ('Structural signals use only prior chronological rows and are investigation leads, '
                        'not proof of laundering or a substitute for the IBM AML model.')}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rows', type=int, default=MAX_ROWS_DEFAULT)
    parser.add_argument('--rapid-seconds', type=int, default=900)
    parser.add_argument('--min-degree', type=int, default=3)
    parser.add_argument('--max-results', type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Output exists; preserve prior graph reports: {args.output}')
    report = analyze(args.data, args.max_rows, args.rapid_seconds, args.min_degree, args.max_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'rows_analyzed': report['rows_analyzed'],
                      'rapid_chains': len(report['rapid_transfer_chains']),
                      'cycles': len(report['directed_three_node_cycles'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
