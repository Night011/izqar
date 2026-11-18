import numpy as np
import pytest

from app.unified_network import TASKS, UnifiedNetwork, required_shapes


def tiny_weights():
    dimensions = {task: 2 for task in TASKS}
    weights = {name: np.zeros(shape, dtype=np.float32)
               for name, shape in required_shapes(dimensions).items()}
    for task in TASKS:
        weights[f'encoders.{task}.weight'][0, 0] = 1
        weights[f'heads.{task}.weight'][0, 0] = 1
    weights['shared.0.weight'][0, 0] = 1
    weights['shared.2.weight'][0, 0] = 1
    return dimensions, weights


def test_four_heads_pass_through_same_shared_weights():
    dimensions, weights = tiny_weights()
    model = UnifiedNetwork(weights, dimensions)
    x = np.array([[0, 0], [2, 0]], dtype=np.float32)
    first = model.predict_scores('paysim', x)
    assert first[0] == pytest.approx(.5)
    assert first[1] == pytest.approx(1 / (1 + np.exp(-2)))
    assert np.array_equal(first, model.predict_scores('ibm_aml', x))
    weights['shared.2.weight'][0, 0] = 2
    changed = UnifiedNetwork(weights, dimensions)
    for task in TASKS:
        assert changed.predict_scores(task, x)[1] > model.predict_scores(task, x)[1]


def test_rejects_missing_task_nonfinite_weights_and_pickle(tmp_path):
    dimensions, weights = tiny_weights()
    with pytest.raises(ValueError, match='exactly four'):
        required_shapes({'paysim': 2})
    with pytest.raises(ValueError, match='weight keys mismatch'):
        UnifiedNetwork({k: v for k, v in weights.items() if k != 'shared.0.bias'}, dimensions)
    weights['shared.0.bias'][0] = np.nan
    with pytest.raises(ValueError, match='finite float32'):
        UnifiedNetwork(weights, dimensions)
    weights['shared.0.bias'][0] = 0
    path = tmp_path / 'weights.npz'
    np.savez_compressed(path, **weights)
    assert UnifiedNetwork.from_npz(path, dimensions).predict_scores('baf_base', [[1, 0]]).shape == (1,)
    with pytest.raises(ValueError, match='expects a 2D matrix'):
        UnifiedNetwork.from_npz(path, dimensions).predict_scores('baf_base', [1, 0])
