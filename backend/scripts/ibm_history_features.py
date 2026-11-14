"""Leakage-safe, account-keyed historical features for IBM AML candidates.

Account identifiers are state keys only and never become model columns.  A row
sees state strictly before its timestamp; rows sharing a timestamp are scored
as a batch before any of them update history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


HISTORY_FEATURES = [
    'sender_prior_outgoing_count', 'recipient_prior_incoming_count',
    'sender_prior_outgoing_amount', 'recipient_prior_incoming_amount',
    'seconds_since_sender_outgoing', 'seconds_since_recipient_incoming',
]


class CausalAccountHistory:
    """Bounded account history state; unseen accounts are safely cold-started."""

    def __init__(self, max_accounts: int = 500_000):
        if max_accounts < 1:
            raise ValueError('max_accounts must be positive')
        self.max_accounts = max_accounts
        self._state: dict[str, dict] = {}

    def _entry(self, account: str) -> dict | None:
        existing = self._state.get(account)
        if existing is not None:
            return existing
        if len(self._state) >= self.max_accounts:
            return None
        value = {'out_count': 0, 'in_count': 0, 'out_amount': 0.0, 'in_amount': 0.0,
                 'last_out': None, 'last_in': None}
        self._state[account] = value
        return value

    @staticmethod
    def _seconds(now: pd.Timestamp, earlier: pd.Timestamp | None) -> float:
        return -1.0 if earlier is None else max(0.0, float((now - earlier).total_seconds()))

    def features(self, at: pd.Timestamp, sender: str, recipient: str) -> dict:
        source, target = self._state.get(sender), self._state.get(recipient)
        return {
            'sender_prior_outgoing_count': 0.0 if source is None else float(source['out_count']),
            'recipient_prior_incoming_count': 0.0 if target is None else float(target['in_count']),
            'sender_prior_outgoing_amount': 0.0 if source is None else float(source['out_amount']),
            'recipient_prior_incoming_amount': 0.0 if target is None else float(target['in_amount']),
            'seconds_since_sender_outgoing': -1.0 if source is None else self._seconds(at, source['last_out']),
            'seconds_since_recipient_incoming': -1.0 if target is None else self._seconds(at, target['last_in']),
        }

    def record(self, at: pd.Timestamp, sender: str, recipient: str, amount: float) -> None:
        if not np.isfinite(amount) or amount < 0:
            raise ValueError('amount must be finite and nonnegative')
        source, target = self._entry(sender), self._entry(recipient)
        if source is not None:
            source['out_count'] += 1; source['out_amount'] += amount; source['last_out'] = at
        if target is not None:
            target['in_count'] += 1; target['in_amount'] += amount; target['last_in'] = at

    @property
    def tracked_accounts(self) -> int:
        return len(self._state)


class CausalHistoryStream:
    """Chunk adapter that never leaks a timestamp group across chunk boundaries."""

    def __init__(self, history: CausalAccountHistory | None = None):
        self.history = history or CausalAccountHistory()
        self._carry: pd.DataFrame | None = None

    def feed(self, chunk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if chunk.empty:
            return chunk.copy(), pd.DataFrame(columns=HISTORY_FEATURES, dtype='float32')
        required = {'Timestamp', 'Account', 'Account.1', 'Amount Paid'}
        if missing := required - set(chunk.columns):
            raise ValueError(f'missing historical feature columns: {sorted(missing)}')
        times = pd.to_datetime(chunk['Timestamp'], errors='raise')
        if times.isna().any() or not times.is_monotonic_increasing:
            raise ValueError('history stream requires chronological, complete timestamps')
        combined = pd.concat([self._carry, chunk], ignore_index=True) if self._carry is not None else chunk.copy()
        all_times = pd.to_datetime(combined['Timestamp'], errors='raise')
        last_time = all_times.iloc[-1]
        ready = combined.loc[all_times < last_time].copy()
        self._carry = combined.loc[all_times == last_time].copy()
        if ready.empty:
            return ready, pd.DataFrame(columns=HISTORY_FEATURES, index=ready.index, dtype='float32')
        features, self.history = build_history_features(ready, self.history)
        return ready, features

    def finish(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._carry is None or self._carry.empty:
            return pd.DataFrame(), pd.DataFrame(columns=HISTORY_FEATURES, dtype='float32')
        ready, self._carry = self._carry, None
        features, self.history = build_history_features(ready, self.history)
        return ready, features


def build_history_features(rows: pd.DataFrame, history: CausalAccountHistory | None = None) -> tuple[pd.DataFrame, CausalAccountHistory]:
    """Produce features for a chronologically sorted frame without labels.

    Required columns deliberately use the disambiguated pandas names for the
    duplicate Account columns in IBM's CSV: ``Account`` and ``Account.1``.
    """
    required = {'Timestamp', 'Account', 'Account.1', 'Amount Paid'}
    if missing := required - set(rows.columns):
        raise ValueError(f'missing historical feature columns: {sorted(missing)}')
    current = history or CausalAccountHistory()
    times = pd.to_datetime(rows['Timestamp'], errors='raise')
    if times.isna().any() or not times.is_monotonic_increasing:
        raise ValueError('history features require chronological, complete timestamps')
    amounts = pd.to_numeric(rows['Amount Paid'], errors='coerce')
    if amounts.isna().any() or not np.isfinite(amounts.to_numpy()).all() or (amounts < 0).any():
        raise ValueError('Amount Paid must be finite and nonnegative')
    senders = rows['Account'].astype('string')
    recipients = rows['Account.1'].astype('string')
    if senders.isna().any() or recipients.isna().any() or (senders.str.len() == 0).any() or (recipients.str.len() == 0).any():
        raise ValueError('account identifiers must be nonempty for history construction')

    output: list[dict] = [None] * len(rows)  # type: ignore[list-item]
    start = 0
    while start < len(rows):
        end = start + 1
        while end < len(rows) and times.iloc[end] == times.iloc[start]:
            end += 1
        # Query all simultaneous rows before mutating state: no same-timestamp leakage.
        for index in range(start, end):
            output[index] = current.features(times.iloc[index], str(senders.iloc[index]), str(recipients.iloc[index]))
        for index in range(start, end):
            current.record(times.iloc[index], str(senders.iloc[index]), str(recipients.iloc[index]), float(amounts.iloc[index]))
        start = end
    return pd.DataFrame(output, columns=HISTORY_FEATURES, index=rows.index, dtype='float32'), current
