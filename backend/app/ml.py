"""Trusted, separately versioned fraud models and strict inference schemas."""
from functools import lru_cache
import hashlib
from pathlib import Path
import json
import joblib
import sklearn
from .config import MODEL_PATH, PAYSIM_MODEL_PATH, BAF_MODEL_PATH, IBM_AML_MODEL_PATH, UNIFIED_MODEL_PATH
from .model_registry import FORBIDDEN as FORBIDDEN_MODEL_FEATURES
from .unified_network import TASKS as UNIFIED_TASKS, UnifiedNetwork

PAYSIM_RAW_FEATURES = ('type', 'amount', 'oldbalanceOrg', 'oldbalanceDest')
PAYSIM_V2_FEATURES = PAYSIM_RAW_FEATURES + ('amount_to_origin_balance', 'amount_to_destination_balance',
                                            'destination_balance_is_zero')
REVIEWED_PAYSIM_V5_LEGACY_SKLEARN = '1.6.1'


def _is_reviewed_paysim_v5_legacy(report: dict, expected_kind: str, portable: bool) -> bool:
    """Allow only the reviewed portable v5 preprocessing migration.

    This is not a general sklearn pickle compatibility promise.  The source
    artifact is a sealed PaySim v5 export, and its direct transform has been
    verified locally after restoring the derived imputer cache.
    """
    return (portable and expected_kind == 'paysim' and report.get('model_version') == 'paysim-xgboost-v5'
            and tuple(report.get('features', ())) == PAYSIM_V2_FEATURES
            and report.get('library_versions', {}).get('scikit_learn') == REVIEWED_PAYSIM_V5_LEGACY_SKLEARN
            and sklearn.__version__ == '1.9.1')


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _local_xgboost_explanation(artifact: dict, frame) -> list[dict]:
    """Return bounded, local TreeSHAP log-odds contributions for one record.

    Feature values are never echoed back.  Contributions describe the fitted
    model, not causal relationships or a calibrated fraud probability.  The
    explanation is deliberately best-effort: an inference response remains
    available if a legacy trusted artifact cannot expose feature names.
    """
    try:
        import math
        import numpy as np
        from xgboost import DMatrix

        pipeline = artifact['pipeline']
        preprocessing = pipeline.named_steps['preprocessing']
        model = pipeline.named_steps['model']
        transformed = preprocessing.transform(frame)
        names = list(preprocessing.get_feature_names_out())
        contributions = model.get_booster().predict(DMatrix(transformed), pred_contribs=True)[0]
        if len(names) != len(contributions) - 1:
            return []
        ranked = sorted(
            ((str(name), float(value)) for name, value in zip(names, contributions[:-1])
             if math.isfinite(float(value))),
            key=lambda item: (-abs(item[1]), item[0]),
        )[:5]
        return [{'feature': name, 'log_odds_contribution': round(value, 6),
                 'direction': 'increases_model_score' if value > 0 else 'decreases_model_score'}
                for name, value in ranked]
    except Exception:
        return []


def _prepare_legacy_unified_sklearn_types() -> bool:
    """Restore the harmless list subclass removed after scikit-learn 1.6.

    The reviewed Unified run was created with scikit-learn 1.6.1. Its
    ``ColumnTransformer`` pickle names this private list subclass, while the
    local runtime intentionally stays on 1.9.1 for specialist artifacts.
    The model uses ``remainder='drop'``; recreating the list container lets
    the trusted pickle deserialize without changing features.
    """
    from sklearn.compose import _column_transformer as column_transformer
    if hasattr(column_transformer, '_RemainderColsList'):
        return False

    class _RemainderColsList(list):
        pass

    column_transformer._RemainderColsList = _RemainderColsList
    return True


def _repair_legacy_unified_preprocessors(preprocessors) -> bool:
    """Fill the renamed SimpleImputer cache in reviewed sklearn 1.6 exports.

    Older exports can place the imputer directly in a ``ColumnTransformer``
    or inside a pipeline.  Walk only fitted transformer containers; this
    repair restores a derived runtime cache and never changes statistics,
    feature names, or learned model weights.
    """
    from sklearn.impute import SimpleImputer
    repaired = False
    visited: set[int] = set()

    def repair(transformer) -> None:
        nonlocal repaired
        if id(transformer) in visited:
            return
        visited.add(id(transformer))
        if (isinstance(transformer, SimpleImputer) and not hasattr(transformer, '_fill_dtype')
                and hasattr(transformer, '_fit_dtype')):
            transformer._fill_dtype = transformer._fit_dtype
            repaired = True
        if hasattr(transformer, 'named_steps'):
            for step in transformer.named_steps.values():
                repair(step)
        if hasattr(transformer, 'transformers_'):
            for _, child, _ in transformer.transformers_:
                repair(child)

    for transformer in preprocessors.values():
        repair(transformer)
    return repaired


def _load_unified_preprocessors(path: Path):
    """Load the trusted preprocessing export, including the reviewed 1.6 shim."""
    import warnings
    from sklearn.exceptions import InconsistentVersionWarning

    legacy_list = _prepare_legacy_unified_sklearn_types()
    # The mismatch is exposed in model status. Avoid logging it for every
    # process that loads the cached artifact.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', InconsistentVersionWarning)
        preprocessors = joblib.load(path)
    repaired = _repair_legacy_unified_preprocessors(preprocessors)
    return preprocessors, legacy_list or repaired


def _load_unified_final_evaluation(path: Path, report: dict, weights: Path, preprocessors: Path):
    """Read an optional final report only when it matches the serving export."""
    final_path = path.with_name(path.name.removesuffix('.metrics.json') + '.final_test.json')
    if not final_path.is_file():
        return None, None
    try:
        final = json.loads(final_path.read_text(encoding='utf-8'))
        if (final.get('format_version') != 1 or final.get('evaluation_type') != 'one_time_sealed_final_test'
                or final.get('model_version') != report.get('model_version')
                or final.get('manifest_sha256') != _sha256_file(path)
                or final.get('weights_sha256') != _sha256_file(weights)
                or final.get('preprocessors_sha256') != _sha256_file(preprocessors)):
            raise ValueError('provenance does not match the installed export')
        tasks = final.get('tasks')
        if not isinstance(tasks, dict) or set(tasks) != set(UNIFIED_TASKS):
            raise ValueError('missing final metrics for one or more tasks')
        for task in UNIFIED_TASKS:
            item = tasks[task]
            threshold = item.get('threshold_from_validation') if isinstance(item, dict) else None
            metrics = item.get('final_test_metrics') if isinstance(item, dict) else None
            if (not isinstance(metrics, dict) or threshold != report['tasks'][task].get('threshold')
                    or not {'precision', 'recall', 'f1', 'average_precision_pr_auc', 'roc_auc',
                            'confusion_matrix_tn_fp_fn_tp', 'false_positive_rate', 'false_negative_rate',
                            'false_alerts_per_10000_operations', 'cases'}.issubset(metrics)):
                raise ValueError(f'invalid final metrics for {task}')
        return final, None
    except (OSError, ValueError, TypeError) as exc:
        return None, f'Final evaluation report was ignored: {type(exc).__name__}'


def _load_artifact(path: Path, expected_kind: str):
    if not path.is_file():
        return None
    # A sidecar is readable without executing pickle code. Reject known version
    # mismatches before joblib.load; sklearn does not support cross-version loads.
    portable = path.name.endswith('.metrics.json')
    report_path = path if portable else path.with_suffix('.metrics.json')
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f'Invalid {expected_kind} model metadata') from exc
        if report.get('model_kind') != expected_kind:
            raise RuntimeError(f'Incorrect {expected_kind} model kind in metadata')
        trained_version = report.get('library_versions', {}).get('scikit_learn')
        legacy_paysim_v5 = _is_reviewed_paysim_v5_legacy(report, expected_kind, portable)
        if trained_version and trained_version != sklearn.__version__ and not legacy_paysim_v5:
            raise RuntimeError(
                f'{expected_kind} requires scikit-learn {trained_version}; runtime has {sklearn.__version__}. '
                'Use a compatible artifact or retrain with matching versions.'
            )
    try:
        if portable:
            if not report_path.is_file() or report.get('format_version') != 3 or report.get('serialization') != 'portable_xgboost_v1':
                raise ValueError('Unsupported portable manifest')
            from sklearn.pipeline import Pipeline
            from xgboost import XGBClassifier
            stem = path.name.removesuffix('.metrics.json')
            if legacy_paysim_v5:
                import warnings
                from sklearn.exceptions import InconsistentVersionWarning
                _prepare_legacy_unified_sklearn_types()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', InconsistentVersionWarning)
                    preprocessing = joblib.load(path.with_name(stem + '.preprocessor.joblib'))
                _repair_legacy_unified_preprocessors({'paysim': preprocessing})
            else:
                preprocessing = joblib.load(path.with_name(stem + '.preprocessor.joblib'))
            model = XGBClassifier()
            model.load_model(path.with_name(stem + '.xgboost.json'))
            artifact = dict(report)
            artifact['pipeline'] = Pipeline([('preprocessing', preprocessing), ('model', model)])
            artifact['preprocessing_compatibility'] = ('reviewed_sklearn_1_6_to_1_9_adapter'
                                                        if legacy_paysim_v5 else None)
        else:
            artifact = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f'Unable to load {expected_kind} model artifact ({type(exc).__name__})') from exc
    if (not isinstance(artifact, dict) or artifact.get('format_version') != (3 if portable else 2)
            or artifact.get('model_kind') != expected_kind or not artifact.get('features')
            or 'pipeline' not in artifact or 'threshold' not in artifact):
        raise RuntimeError(f'Unsupported {expected_kind} model artifact format')
    trained_version = artifact.get('library_versions', {}).get('scikit_learn')
    if trained_version and trained_version != sklearn.__version__ and not _is_reviewed_paysim_v5_legacy(
            artifact, expected_kind, portable):
        raise RuntimeError(f'{expected_kind} artifact requires scikit-learn {trained_version}; runtime has {sklearn.__version__}')
    return artifact


@lru_cache(maxsize=1)
def load_model():
    return _load_artifact(Path(MODEL_PATH), 'ieee_cis')


@lru_cache(maxsize=1)
def load_paysim_model():
    return _load_artifact(PAYSIM_MODEL_PATH, 'paysim')


@lru_cache(maxsize=1)
def load_baf_model():
    return _load_artifact(BAF_MODEL_PATH, 'baf_base')


@lru_cache(maxsize=1)
def load_ibm_aml_model():
    return _load_artifact(IBM_AML_MODEL_PATH, 'ibm_aml')


@lru_cache(maxsize=1)
def load_unified_model():
    """Load a trusted Kaggle export only after strict manifest validation."""
    path = Path(UNIFIED_MODEL_PATH)
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
        if (report.get('format_version') != 1 or report.get('model_kind') != 'izqar_unified_mtl'
                or report.get('serialization') != 'numpy_shared_backbone_v1'):
            raise ValueError('unsupported manifest')
        task_metadata = report.get('task_metadata', {})
        dimensions = report.get('input_dimensions')
        if set(task_metadata) != set(UNIFIED_TASKS) or not isinstance(dimensions, dict):
            raise ValueError('all four task adapters are required')
        schema = {}
        for task in UNIFIED_TASKS:
            item = task_metadata[task].get('feature_schema', {})
            all_features, numerical, categorical = item.get('all'), item.get('numerical'), item.get('categorical')
            if (not isinstance(all_features, list) or not all_features or not isinstance(numerical, list)
                    or not isinstance(categorical, list) or set(all_features) != set(numerical) | set(categorical)
                    or len(all_features) != len(set(all_features)) or len(numerical) != len(set(numerical))
                    or len(categorical) != len(set(categorical)) or set(all_features) & FORBIDDEN_MODEL_FEATURES):
                raise ValueError(f'invalid {task} input schema')
            schema[task] = {'all': all_features, 'numerical': numerical, 'categorical': categorical}
        stem = path.name.removesuffix('.metrics.json')
        weights = path.with_name(stem + '_weights.npz')
        preprocessors_path = path.with_name(stem + '_preprocessors.joblib')
        if not weights.is_file() or not preprocessors_path.is_file():
            raise ValueError('unified model component missing')
        preprocessors, preprocessing_compatibility = _load_unified_preprocessors(preprocessors_path)
        if set(preprocessors) != set(UNIFIED_TASKS):
            raise ValueError('preprocessor task mismatch')
        network = UnifiedNetwork.from_npz(weights, dimensions)
        final_evaluation, final_evaluation_error = _load_unified_final_evaluation(path, report, weights, preprocessors_path)
        return {'report': report, 'network': network, 'preprocessors': preprocessors, 'schema': schema,
                'preprocessing_compatibility': ('sklearn_1_6_to_1_9_adapter' if preprocessing_compatibility else None),
                'final_evaluation': final_evaluation, 'final_evaluation_error': final_evaluation_error}
    except Exception as exc:
        raise RuntimeError(f'Unable to load unified model artifact ({type(exc).__name__})') from exc


def predict_ieee(features: dict):
    artifact = load_model()
    if artifact is None:
        return None
    expected = artifact['features']
    missing = sorted(set(expected) - set(features))
    unknown = sorted(set(features) - set(expected))
    if missing or unknown:
        raise ValueError(f'IEEE-CIS schema mismatch; missing: {missing}; unsupported: {unknown}')
    record = {key: features[key] for key in expected}
    import pandas as pd
    frame = pd.DataFrame([record], columns=expected)
    for col in artifact['numerical_features']:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
        if features[col] is not None and pd.isna(frame[col].iloc[0]):
            raise ValueError(f'{col} must be numeric or null')
    for col in artifact['categorical_features']:
        frame[col] = frame[col].astype('string').fillna('__missing__').astype(str)
    score = float(artifact['pipeline'].predict_proba(frame)[0, 1])
    return {'risk_score': round(score * 100, 2), 'threshold': artifact['threshold'],
            'flagged': score >= artifact['threshold'],
            'source': 'ieee_cis_ml', 'model_version': artifact.get('model_version', 'unknown'),
            'score_kind': 'uncalibrated_ml_score',
            'explanation_method': 'local_tree_shap_log_odds',
            'explanation': _local_xgboost_explanation(artifact, frame),
            'warning': 'Результат модели на исследовательских данных, не подтверждённая вероятность мошенничества в банке.'}


def predict_paysim(features: dict):
    artifact = load_paysim_model()
    if artifact is None:
        return None
    expected = tuple(artifact['features'])
    if expected not in (PAYSIM_RAW_FEATURES, PAYSIM_V2_FEATURES):
        raise ValueError('Unsupported PaySim artifact feature schema')
    missing = [key for key in PAYSIM_RAW_FEATURES if key not in features]
    unknown = sorted(set(features) - set(PAYSIM_RAW_FEATURES))
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unsupported fields: {', '.join(unknown)}")
        raise ValueError('; '.join(detail))
    import pandas as pd
    frame = pd.DataFrame([{key: features[key] for key in PAYSIM_RAW_FEATURES}], columns=PAYSIM_RAW_FEATURES)
    frame['amount'] = pd.to_numeric(frame['amount'], errors='coerce')
    frame['oldbalanceOrg'] = pd.to_numeric(frame['oldbalanceOrg'], errors='coerce')
    frame['oldbalanceDest'] = pd.to_numeric(frame['oldbalanceDest'], errors='coerce')
    if frame[['amount', 'oldbalanceOrg', 'oldbalanceDest']].isna().any().any() or (frame[['amount', 'oldbalanceOrg', 'oldbalanceDest']] < 0).any().any():
        raise ValueError('amount, oldbalanceOrg and oldbalanceDest must be non-negative numbers')
    if expected == PAYSIM_V2_FEATURES:
        # These are pure functions of values supplied before the decision. Do
        # not add ``oldbalanceOrg - amount``: on PaySim it proxies the banned
        # observed post-event balance.
        frame['amount_to_origin_balance'] = frame['amount'] / frame['oldbalanceOrg'].clip(lower=1.0)
        frame['amount_to_destination_balance'] = frame['amount'] / frame['oldbalanceDest'].clip(lower=1.0)
        frame['destination_balance_is_zero'] = (frame['oldbalanceDest'] == 0).astype('float32')
        frame = frame.loc[:, expected]
    score = float(artifact['pipeline'].predict_proba(frame)[0, 1])
    return {'risk_score': round(score * 100, 2), 'threshold': artifact['threshold'],
            'flagged': score >= artifact['threshold'], 'source': 'paysim_ml',
            'model_version': artifact.get('model_version', 'unknown'),
            'score_kind': 'uncalibrated_ml_score',
            'explanation_method': 'local_tree_shap_log_odds',
            'explanation': _local_xgboost_explanation(artifact, frame),
            'warning': 'PaySim is synthetic. This ML score is not a calibrated banking fraud probability.'}


def predict_baf(features: dict):
    artifact = load_baf_model()
    if artifact is None:
        return None
    expected = artifact['features']
    missing = sorted(set(expected) - set(features))
    unknown = sorted(set(features) - set(expected))
    if missing or unknown:
        raise ValueError(f'BAF Base schema mismatch; missing: {missing}; unsupported: {unknown}')
    import pandas as pd
    frame = pd.DataFrame([{name: features[name] for name in expected}], columns=expected)
    for col in artifact['numerical_features']:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
        if features[col] is not None and pd.isna(frame[col].iloc[0]):
            raise ValueError(f'{col} must be numeric or null')
    for col in ('prev_address_months_count', 'current_address_months_count',
                'bank_months_count', 'session_length_in_minutes'):
        frame[col] = frame[col].mask(frame[col] == -1)
    for col in artifact['categorical_features']:
        if features[col] is not None and not isinstance(features[col], str):
            raise ValueError(f'{col} must be text or null')
        frame[col] = frame[col].astype('string').fillna('__missing__').astype(str)
    score = float(artifact['pipeline'].predict_proba(frame)[0, 1])
    return {'risk_score': round(score * 100, 2), 'threshold': artifact['threshold'],
            'flagged': score >= artifact['threshold'], 'source': 'baf_base_ml',
            'model_version': artifact.get('model_version', 'unknown'),
            'score_kind': 'uncalibrated_ml_score',
            'explanation_method': 'local_tree_shap_log_odds',
            'explanation': _local_xgboost_explanation(artifact, frame),
            'warning': 'BAF Base is synthetic account-application data, not a calibrated fraud probability for a bank.'}


def predict_ibm_aml(features: dict):
    artifact = load_ibm_aml_model()
    if artifact is None:
        return None
    expected = artifact['features']
    missing = sorted(set(expected) - set(features))
    unknown = sorted(set(features) - set(expected))
    if missing or unknown:
        raise ValueError(f'IBM AML schema mismatch; missing: {missing}; unsupported: {unknown}')
    import math
    import pandas as pd
    frame = pd.DataFrame([{name: features[name] for name in expected}], columns=expected)
    for col in artifact['numerical_features']:
        if features[col] is None or isinstance(features[col], bool):
            raise ValueError(f'{col} must be a finite number')
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
        if not math.isfinite(float(frame[col].iloc[0])):
            raise ValueError(f'{col} must be a finite number')
    if frame['amount_paid'].iloc[0] < 0 or abs(float(frame['log_amount_paid'].iloc[0]) - math.log1p(float(frame['amount_paid'].iloc[0]))) > 1e-4:
        raise ValueError('amount_paid must be nonnegative and log_amount_paid must match it')
    if any(frame[name].iloc[0] not in (0, 1) for name in ('same_bank', 'same_currency')):
        raise ValueError('same_bank and same_currency must be 0 or 1')
    if not 0 <= frame['hour'].iloc[0] <= 23 or not 0 <= frame['weekday'].iloc[0] <= 6:
        raise ValueError('hour must be 0–23 and weekday 0–6')
    for col in artifact['categorical_features']:
        if not isinstance(features[col], str) or not features[col].strip():
            raise ValueError(f'{col} must be nonempty text')
        frame[col] = frame[col].astype(str)
    if frame['same_currency'].iloc[0] != int(features['payment_currency'] == features['receiving_currency']):
        raise ValueError('same_currency must match payment_currency and receiving_currency')
    score = float(artifact['pipeline'].predict_proba(frame)[0, 1])
    return {'risk_score': round(score * 100, 2), 'threshold': artifact['threshold'],
            'flagged': score >= artifact['threshold'], 'source': 'ibm_aml_ml',
            'model_version': artifact.get('model_version', 'unknown'),
            'score_kind': 'uncalibrated_ml_score',
            'explanation_method': 'local_tree_shap_log_odds',
            'explanation': _local_xgboost_explanation(artifact, frame),
            'warning': 'IBM AML uses synthetic laundering labels; this is not a calibrated bank fraud probability.'}


def predict_unified(task: str, features: dict):
    artifact = load_unified_model()
    if artifact is None:
        return None
    if task not in UNIFIED_TASKS:
        raise ValueError('Unknown unified task')
    schema = artifact['schema'][task]
    expected = schema['all']
    missing, unknown = sorted(set(expected) - set(features)), sorted(set(features) - set(expected))
    if missing or unknown:
        raise ValueError(f'Unified {task} schema mismatch; missing: {missing}; unsupported: {unknown}')
    import pandas as pd
    frame = pd.DataFrame([{name: features[name] for name in expected}], columns=expected)
    for name in schema['numerical']:
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
        if features[name] is not None and pd.isna(frame[name].iloc[0]):
            raise ValueError(f'{name} must be numeric or null')
    for name in schema['categorical']:
        if features[name] is not None and not isinstance(features[name], str):
            raise ValueError(f'{name} must be text or null')
    transformed = artifact['preprocessors'][task].transform(frame)
    score = float(artifact['network'].predict_scores(task, transformed)[0])
    threshold = artifact['report']['tasks'][task]['threshold']
    return {'risk_score': round(score * 100, 2), 'threshold': threshold, 'flagged': score >= threshold,
            'source': 'izqar_unified_mtl_experimental', 'model_version': artifact['report']['model_version'],
            'score_kind': 'uncalibrated_ml_score',
            'explanation_method': 'not_available_for_experimental_shared_backbone', 'explanation': [],
            'warning': 'Experimental joint model. This is not a calibrated probability and does not replace the specialised detector unless validation comparison supports it.'}
