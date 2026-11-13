"""A small trusted export proves the Unified CPU inference contract end to end."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import FunctionTransformer

from app.unified_network import TASKS, required_shapes


def make_artifact(root: Path):
    report = {'format_version': 1, 'model_kind': 'izqar_unified_mtl',
              'model_version': 'izqar-unified-mtl-test', 'serialization': 'numpy_shared_backbone_v1',
              'input_dimensions': {task: 1 for task in TASKS},
              'tasks': {task: {'threshold': .5, 'validation_metrics': {'f1': .1}} for task in TASKS},
              'task_metadata': {task: {'feature_schema': {'all': ['amount'], 'numerical': ['amount'], 'categorical': []}}
                                for task in TASKS}}
    manifest = root / 'unified.metrics.json'
    manifest.write_text(json.dumps(report), encoding='utf-8')
    preprocessors = {task: FunctionTransformer(validate=False) for task in TASKS}
    joblib.dump(preprocessors, root / 'unified_preprocessors.joblib')
    weights = {key: np.zeros(shape, dtype=np.float32)
               for key, shape in required_shapes(report['input_dimensions']).items()}
    np.savez_compressed(root / 'unified_weights.npz', **weights)
    return manifest


def test_unified_missing_or_valid_cpu_artifact(client, auth, tmp_path, monkeypatch):
    import app.ml as ml
    missing = tmp_path / 'missing.metrics.json'
    with monkeypatch.context() as patch:
        patch.setattr(ml, 'UNIFIED_MODEL_PATH', missing)
        ml.load_unified_model.cache_clear()
        assert client.post('/api/v1/unified/paysim/predict', headers=auth,
                           json={'features': {'amount': 10}}).status_code == 503
        manifest = make_artifact(tmp_path)
        patch.setattr(ml, 'UNIFIED_MODEL_PATH', manifest)
        ml.load_unified_model.cache_clear()
        response = client.post('/api/v1/unified/ibm_aml/predict', headers=auth,
                               json={'features': {'amount': 10}})
        assert response.status_code == 200, response.text
        assert response.json()['source'] == 'izqar_unified_mtl_experimental'
        assert response.json()['risk_score'] == 50
        assert client.post('/api/v1/unified/unknown/predict', headers=auth,
                           json={'features': {'amount': 10}}).status_code == 422
        assert client.post('/api/v1/unified/paysim/predict', headers=auth,
                           json={'features': {'isFraud': 1}}).status_code == 422
    ml.load_unified_model.cache_clear()
