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
        from app.ml import _sha256_file
        metric = {'precision': .1, 'recall': .2, 'f1': .133333, 'average_precision_pr_auc': .12,
                  'roc_auc': .55, 'confusion_matrix_tn_fp_fn_tp': [8, 1, 2, 1],
                  'false_positive_rate': .111111, 'false_negative_rate': .666667,
                  'false_alerts_per_10000_operations': 833.333, 'cases': 12, 'positive_cases': 3}
        final = {'format_version': 1, 'evaluation_type': 'one_time_sealed_final_test',
                 'model_version': 'izqar-unified-mtl-test',
                 'manifest_sha256': _sha256_file(manifest),
                 'weights_sha256': _sha256_file(tmp_path / 'unified_weights.npz'),
                 'preprocessors_sha256': _sha256_file(tmp_path / 'unified_preprocessors.joblib'),
                 'tasks': {task: {'threshold_from_validation': .5, 'final_test_metrics': metric} for task in TASKS}}
        (tmp_path / 'unified.final_test.json').write_text(json.dumps(final), encoding='utf-8')
        ml.load_unified_model.cache_clear()
        status = client.get('/api/v1/model/status', headers=auth)
        assert status.status_code == 200
        assert status.json()['unified_mtl']['task_schemas'] == {task: ['amount'] for task in TASKS}
        assert status.json()['unified_mtl']['final_evaluation_available'] is True
        assert status.json()['unified_mtl']['tasks']['paysim']['final_test_metrics']['cases'] == 12
        (tmp_path / 'unified.final_test.json').write_text('{}', encoding='utf-8')
        ml.load_unified_model.cache_clear()
        ignored = client.get('/api/v1/model/status', headers=auth).json()['unified_mtl']
        assert ignored['final_evaluation_available'] is False
        assert ignored['final_evaluation_error']
        assert client.post('/api/v1/unified/paysim/predict', headers=auth,
                           json={'features': {'amount': 10}}).status_code == 200
    ml.load_unified_model.cache_clear()
