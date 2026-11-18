"""Score a batch of IEEE-like transactions. Reproduces causal frequency features exactly.

Rows are scored in TransactionDT order (stable) so every freq feature equals the
training-time definition: the number of prior rows with the same key.
Output is written in the ORIGINAL input row order.

Usage:
    python backend/scripts/score_ieee_batch.py --model path/to/model.joblib --input batch.csv --output scored.csv

The model artifact must carry format_version=1, pipeline, features,
num/categorical feature lists and (for freq artifacts) freq_sources/freq_scale.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Score an IEEE-CIS batch (causal freq reproduction)')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    if artifact.get('format_version') != 1 or 'pipeline' not in artifact:
        raise SystemExit('Unsupported artifact: expected format_version=1 with pipeline')
    features = artifact['features']
    nums = artifact['numerical_features']
    cats = artifact['categorical_features']
    data = pd.read_csv(args.input, low_memory=False)
    data['__input_order'] = np.arange(len(data))
    sorted_data = data.sort_values('TransactionDT', kind='stable').reset_index(drop=True) \
        if 'TransactionDT' in data.columns else data.reset_index(drop=True)

    frame = pd.DataFrame({col: sorted_data.get(col) for col in features}, columns=features)
    scale = artifact.get('freq_scale', 'log')
    sources = artifact.get('freq_sources') or {}
    for name, cols in sources.items():
        if name in features and all(col in sorted_data.columns for col in cols):
            key = sorted_data[cols].astype('string').fillna('__missing__')
            key = key.iloc[:, 0] if len(cols) == 1 else key.agg('_'.join, axis=1)
            counts = key.groupby(key).cumcount().astype('float32')
            frame[name] = np.log1p(counts) if scale == 'log' else counts
    for col in nums:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    for col in cats:
        frame[col] = frame[col].astype('string').fillna('__missing__').astype(str)

    scores = artifact['pipeline'].predict_proba(frame)[:, 1]
    sorted_data = sorted_data.copy()
    sorted_data['risk_score'] = np.round(scores * 100, 2)
    sorted_data['flagged'] = scores >= float(artifact['threshold'])
    out = sorted_data.sort_values('__input_order', kind='stable').reset_index(drop=True)
    out = out.drop(columns=['__input_order'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f'scored {len(out)} rows -> {args.output}')


if __name__ == '__main__':
    main()
