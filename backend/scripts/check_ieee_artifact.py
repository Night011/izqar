"""Validate the structure of a local IEEE-CIS model artifact.

Usage:
    python backend/scripts/check_ieee_artifact.py path/to/model.joblib

Checks format_version, required keys, feature lists, and that the pipeline
produces finite probabilities on a synthetic NaN row. Exits non-zero on failure.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REQUIRED_KEYS = {'format_version', 'pipeline', 'features', 'numerical_features',
                 'categorical_features', 'threshold', 'split_method'}

def check(path: Path) -> dict:
    errors: list[str] = []
    if not path.is_file():
        return {'ok': False, 'errors': [f'model not found: {path}']}
    artifact = joblib.load(path)
    missing = REQUIRED_KEYS - set(artifact)
    if missing:
        errors.append(f'missing keys: {sorted(missing)}')
    if artifact.get('format_version') != 1:
        errors.append(f"format_version must be 1, got {artifact.get('format_version')}")
    features = list(artifact.get('features') or [])
    nums = list(artifact.get('numerical_features') or [])
    cats = list(artifact.get('categorical_features') or [])
    if not features:
        errors.append('features list is empty')
    if set(nums) | set(cats) != set(features):
        errors.append('numerical+categorical features do not add up to features')
    if not (0.0 <= float(artifact.get('threshold', -1)) <= 1.0):
        errors.append('threshold outside [0, 1]')
    if not errors:
        freq_sources = artifact.get('freq_sources') or {}
        freq_feats = [f for f in features if f in freq_sources]
        if freq_feats and not artifact.get('freq_state'):
            errors.append(f'freq features {freq_feats} present but freq_state is missing; inference cannot reproduce them')
        row = {key: np.nan for key in features}
        for key in cats:
            row[key] = '__missing__'
        frame = pd.DataFrame([row], columns=features)
        for col in nums:
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
        try:
            score = float(artifact['pipeline'].predict_proba(frame)[0, 1])
            if not np.isfinite(score):
                errors.append('pipeline returned non-finite probability')
        except Exception as exc:  # noqa: BLE001 - report any inference failure
            errors.append(f'pipeline inference failed: {exc}')
    result = {
        'ok': not errors,
        'path': str(path),
        'model_version': artifact.get('model_version'),
        'n_features': len(features),
        'threshold': artifact.get('threshold'),
        'split_method': artifact.get('split_method'),
        'validation_metrics': artifact.get('validation_metrics'),
        'test_metrics': artifact.get('test_metrics'),
        'errors': errors,
    }
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    args = parser.parse_args()
    report = check(args.model)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    sys.exit(0 if report['ok'] else 1)
