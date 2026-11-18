"""Causal frequency features for IEEE-CIS inference.

Two reproduction modes, both deterministic:

* Batch (exact within the supplied batch): rows are sorted by TransactionDT
  (stable) and each row gets the cumcount of prior rows with the same key.
  This reproduces the training-time IEEE features exactly when the scored
  history starts at the same origin; for partial batches it reproduces the
  causal definition within the batch.
* Single record (best effort): when no batch context exists, use
  ``artifact['freq_state']`` — cumulative key counts at the end of the
  training period. Unknown keys map to 0.

Artifacts store ``freq_sources`` (name -> key columns) and optionally
``freq_state`` (name -> {key: train_total}).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FREQ_SOURCES = {
    'card1_freq': ('card1',),
    'card1_card5_freq': ('card1', 'card5'),
    'card2_freq': ('card2',),
    'addr1_freq': ('addr1',),
    'p_email_freq': ('P_emaildomain',),
    'device_freq': ('DeviceInfo',),
}
FREQ_SCALE = 'log'  # artifacts produced with scale='log' use log1p of prior counts


def freq_key(frame: pd.DataFrame, cols) -> pd.Series:
    if len(cols) > 1:
        return frame[list(cols)].astype('string').fillna('__missing__').agg('_'.join, axis=1)
    return frame[cols[0]].astype('string').fillna('__missing__')


def add_freq_features(frame: pd.DataFrame, artifact: dict, scale: str = FREQ_SCALE) -> pd.DataFrame:
    sources = artifact.get('freq_sources', FREQ_SOURCES)
    frame = frame.copy()
    is_batch = 'TransactionDT' in frame.columns and len(frame) > 1
    if is_batch:
        frame = frame.sort_values(['TransactionDT'], kind='stable').reset_index(drop=True)
        for name, cols in sources.items():
            if all(col in frame.columns for col in cols):
                counts = freq_key(frame, cols).groupby(freq_key(frame, cols)).cumcount().astype('float32')
                frame[name] = np.log1p(counts) if scale == 'log' else counts
        return frame
    # Single record / tiny frame: look up cumulative train counts.
    state = artifact.get('freq_state') or {}
    for name, cols in sources.items():
        if all(col in frame.columns for col in cols):
            key = freq_key(frame, cols)
            counts = key.map(state.get(name, {})).fillna(0).astype('float32')
            frame[name] = np.log1p(counts) if scale == 'log' else counts
    return frame


def train_freq_state(train_frame: pd.DataFrame, sources=FREQ_SOURCES) -> dict[str, dict[str, int]]:
    state = {}
    for name, cols in sources.items():
        if any(col not in train_frame.columns for col in cols):
            continue
        key = freq_key(train_frame, cols)
        counts = key.value_counts()
        state[name] = {str(k): int(v) for k, v in counts.items()}
    return state
