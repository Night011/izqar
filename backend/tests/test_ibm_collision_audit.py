"""Collision audit fixture uses public-like synthetic identities only."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_ibm_account_collisions import EXPECTED_HEADER, audit  # noqa: E402


def make_csv(path, *, after='999999'):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(EXPECTED_HEADER)
        writer.writerow(['2025-01-01 00:00:00', '001', 'A', '003', 'B', 10, 'USD', 10, 'USD', 'Wire', 'hidden'])
        writer.writerow(['2025-01-01 00:01:00', '002', 'A', '004', 'C', 5, 'USD', 5, 'USD', 'Wire', 'hidden'])
        writer.writerow(['2025-01-02 00:00:00', '005', 'A', '006', 'Z', 12, 'USD', 12, 'USD', 'Wire', after])


def test_collision_audit_uses_bank_account_pairs_before_boundary(tmp_path):
    path = tmp_path / 'HI-Small_Trans.csv'
    make_csv(path)
    result = audit(path, '2025-01-01 00:02:00')
    assert result['train_rows_scanned'] == 2
    assert result['distinct_account_strings'] == 3
    assert result['distinct_bank_account_pairs'] == 4
    assert result['account_strings_present_in_multiple_banks'] == 1
    assert result['complete_train_window'] is True
    assert result['label_columns_read'] is False
    assert not any(value == 'A' for value in result.values())


def test_future_labels_and_future_account_changes_not_used(tmp_path):
    path = tmp_path / 'HI-Small_Trans.csv'
    make_csv(path, after='not-a-label')
    original = audit(path, '2025-01-01 00:02:00')
    make_csv(path, after='changed-postcutoff')
    modified = audit(path, '2025-01-01 00:02:00')
    assert original == modified


def test_capped_audit_is_explicitly_partial(tmp_path):
    path = tmp_path / 'HI-Small_Trans.csv'
    make_csv(path)
    result = audit(path, '2025-01-01 00:02:00', max_train_rows=1)
    assert result['train_rows_scanned'] == 1
    assert result['complete_train_window'] is False
    assert result['account_strings_present_in_multiple_banks'] == 0


def test_audit_rejects_wrong_dataset_and_empty_window(tmp_path):
    path = tmp_path / 'HI-Small_Trans.csv'
    make_csv(path)
    with pytest.raises(ValueError, match='No train-period'):
        audit(path, '2024-12-31 00:00:00')
    with pytest.raises(ValueError, match='positive'):
        audit(path, '2025-01-02 00:00:00', max_train_rows=0)
    with pytest.raises(ValueError, match='HI-Small'):
        audit(tmp_path / 'anything.csv', '2025-01-02 00:00:00')


def test_collision_audit_rejects_blank_early_identity(tmp_path):
    path = tmp_path / 'HI-Small_Trans.csv'
    make_csv(path)
    raw = path.read_text(encoding='utf-8')
    path.write_text(raw.replace(',001,A,', ',   ,A,'), encoding='utf-8')
    with pytest.raises(ValueError, match='nonempty'):
        audit(path, '2025-01-01 00:02:00')


def test_collision_audit_full_window_is_input_order_independent(tmp_path):
    path = tmp_path / 'HI-Small_Trans.csv'
    make_csv(path)
    expected = audit(path, '2025-01-01 00:02:00')
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.reader(stream))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(rows[0])
        writer.writerows(reversed(rows[1:]))
    actual = audit(path, '2025-01-01 00:02:00')
    assert expected == actual
