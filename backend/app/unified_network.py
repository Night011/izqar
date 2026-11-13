"""Portable CPU inference for the jointly trained IZQAR multi-task network.

The four tasks have distinct input encoders and heads. ``shared.*`` weights
must be produced by a training run in which every task updates them. This
module only evaluates an exported network; it never claims one was trained.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


TASKS = ('paysim', 'ieee_cis', 'baf_base', 'ibm_aml')
HIDDEN = 32
LATENT = 16


def required_shapes(input_dimensions: Mapping[str, int]) -> dict[str, tuple[int, ...]]:
    if set(input_dimensions) != set(TASKS):
        raise ValueError('Unified model requires exactly four task encoders')
    if any(not isinstance(size, int) or size < 1 or size > 20_000
           for size in input_dimensions.values()):
        raise ValueError('Each task input dimension must be between 1 and 20000')
    shapes = {
        'shared.0.weight': (HIDDEN, HIDDEN),
        'shared.0.bias': (HIDDEN,),
        'shared.2.weight': (LATENT, HIDDEN),
        'shared.2.bias': (LATENT,),
    }
    for task in TASKS:
        shapes[f'encoders.{task}.weight'] = (HIDDEN, input_dimensions[task])
        shapes[f'encoders.{task}.bias'] = (HIDDEN,)
        shapes[f'heads.{task}.weight'] = (1, LATENT)
        shapes[f'heads.{task}.bias'] = (1,)
    return shapes


class UnifiedNetwork:
    """Evaluate immutable exported weights; scores are not calibrated probabilities."""

    def __init__(self, weights: Mapping[str, np.ndarray], input_dimensions: Mapping[str, int]):
        expected = required_shapes(input_dimensions)
        if set(weights) != set(expected):
            raise ValueError(f'Unified weight keys mismatch; missing={sorted(set(expected) - set(weights))}; '
                             f'unexpected={sorted(set(weights) - set(expected))}')
        checked = {}
        for name, shape in expected.items():
            value = np.asarray(weights[name])
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f'Invalid unified weight {name}: expected finite float32 {shape}')
            checked[name] = value.copy()
        self.weights = checked
        self.input_dimensions = dict(input_dimensions)

    @classmethod
    def from_npz(cls, path, input_dimensions: Mapping[str, int]) -> 'UnifiedNetwork':
        # NumPy's pickle path is forbidden for model weights.
        with np.load(path, allow_pickle=False) as archive:
            weights = {name: archive[name] for name in archive.files}
        return cls(weights, input_dimensions)

    def predict_scores(self, task: str, features: np.ndarray) -> np.ndarray:
        if task not in TASKS:
            raise ValueError(f'Unknown unified task: {task}')
        if hasattr(features, 'toarray'):
            features = features.toarray()
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dimensions[task]:
            raise ValueError(f'{task} expects a 2D matrix with {self.input_dimensions[task]} columns')
        if not np.isfinite(x).all():
            raise ValueError('Unified input must contain finite transformed values')
        w = self.weights
        hidden = np.maximum(x @ w[f'encoders.{task}.weight'].T + w[f'encoders.{task}.bias'], 0)
        hidden = np.maximum(hidden @ w['shared.0.weight'].T + w['shared.0.bias'], 0)
        hidden = np.maximum(hidden @ w['shared.2.weight'].T + w['shared.2.bias'], 0)
        logits = (hidden @ w[f'heads.{task}.weight'].T + w[f'heads.{task}.bias']).reshape(-1)
        # Stable sigmoid, avoiding overflow from extreme weighted logits.
        positive = logits >= 0
        scores = np.empty(len(logits), dtype=np.float32)
        scores[positive] = 1 / (1 + np.exp(-logits[positive]))
        exp_logit = np.exp(logits[~positive])
        scores[~positive] = exp_logit / (1 + exp_logit)
        return scores
