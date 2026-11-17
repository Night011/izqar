"""Train-period-only IBM AML bank/account identity collision audit.

Never reads the laundering label, does not train models, and returns only
aggregate statistics. Disk-backed SQLite deduplicates account/bank pairs
without retaining a large Python set in RAM.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

EXPECTED_HEADER = [
    'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account',
    'Amount Received', 'Receiving Currency', 'Amount Paid',
    'Payment Currency', 'Payment Format', 'Is Laundering',
]
CHUNK_SIZE = 100_000
COLUMNS = ['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1']


def audit(data_path: Path, train_before: str, max_train_rows: int | None = None) -> dict:
    if data_path.name != 'HI-Small_Trans.csv':
        raise ValueError('Only IBM HI-Small_Trans.csv may be audited')
    with data_path.open('r', encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), None)
    if header != EXPECTED_HEADER:
        raise ValueError('Unexpected IBM AML CSV header; refusing a different dataset')
    boundary = pd.Timestamp(train_before)
    if pd.isna(boundary) or boundary.tzinfo is not None:
        raise ValueError('--train-before must be a valid timezone-naive timestamp')
    if max_train_rows is not None and max_train_rows < 1:
        raise ValueError('--max-train-rows must be positive')
    seen = 0
    truncated = False
    with tempfile.TemporaryDirectory(prefix='izqar-ibm-collision-') as work:
        database = sqlite3.connect(Path(work) / 'accounts.sqlite3')
        try:
            database.execute('PRAGMA journal_mode=OFF')
            database.execute('PRAGMA synchronous=OFF')
            database.execute('CREATE TABLE pairs (account TEXT NOT NULL, bank TEXT NOT NULL, PRIMARY KEY(account, bank)) WITHOUT ROWID')
            for chunk in pd.read_csv(data_path, usecols=COLUMNS,
                                     dtype={'From Bank': 'string', 'To Bank': 'string',
                                            'Account': 'string', 'Account.1': 'string'},
                                     chunksize=CHUNK_SIZE, low_memory=False):
                timestamps = pd.to_datetime(chunk['Timestamp'], errors='raise')
                if timestamps.isna().any() or getattr(timestamps.dt, 'tz', None) is not None:
                    raise ValueError('Timestamps must be complete and timezone-naive')
                earlier = chunk.loc[timestamps < boundary]
                if earlier.empty:
                    continue
                if max_train_rows is not None and seen + len(earlier) > max_train_rows:
                    earlier = earlier.iloc[:max_train_rows - seen]
                    truncated = True
                identifiers = earlier[['From Bank', 'To Bank', 'Account', 'Account.1']]
                missing = identifiers.isna().any().any()
                empty = any(identifiers[col].astype('string').str.strip().eq('').any()
                            for col in identifiers)
                if missing or empty:
                    raise ValueError('Account and bank identifiers must be complete and nonempty')
                origin = list(zip(earlier['Account'].astype(str), earlier['From Bank'].astype(str)))
                recipient = list(zip(earlier['Account.1'].astype(str), earlier['To Bank'].astype(str)))
                database.executemany('INSERT OR IGNORE INTO pairs(account, bank) VALUES (?, ?)', origin)
                database.executemany('INSERT OR IGNORE INTO pairs(account, bank) VALUES (?, ?)', recipient)
                seen += len(earlier)
                if max_train_rows is not None and seen >= max_train_rows:
                    truncated = True
                    break
            pairs = int(database.execute('SELECT COUNT(*) FROM pairs').fetchone()[0])
            identifiers = int(database.execute('SELECT COUNT(DISTINCT account) FROM pairs').fetchone()[0])
            collisions = int(database.execute('SELECT COUNT(*) FROM (SELECT account FROM pairs GROUP BY account HAVING COUNT(*) > 1)').fetchone()[0])
        finally:
            database.close()
    if seen == 0:
        raise ValueError('No train-period operations before the requested timestamp')
    return {
        'audit_kind': 'train_only_ibm_bank_account_collision_counts',
        'source_filename': data_path.name, 'train_before': boundary.isoformat(),
        'train_rows_scanned': seen, 'complete_train_window': not truncated,
        'distinct_bank_account_pairs': pairs,
        'distinct_account_strings': identifiers,
        'account_strings_present_in_multiple_banks': collisions,
        'bank_account_pair_excess_over_global_strings': pairs - identifiers,
        'collision_fraction_of_account_strings': round(collisions / max(identifiers, 1), 6),
        'label_columns_read': False,
        'warning': ('Aggregate identity audit only; no labels, ML training, inference or final-test evaluation. '
                    'If complete_train_window is false, counts are only a lower bound.'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--train-before', required=True,
                        help='Pinned split_boundaries.validation_start from the IBM report')
    parser.add_argument('--max-train-rows', type=int,
                        help='Optional partial diagnostic; omission scans all train rows')
    parser.add_argument('--output', type=Path, help='Optional new JSON file; never overwrite')
    args = parser.parse_args()
    result = audit(args.data, args.train_before, args.max_train_rows)
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(output)
    print(output, end='')


if __name__ == '__main__':
    main()
