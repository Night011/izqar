"""Trusted, separately versioned fraud models and strict inference schemas."""
from functools import lru_cache
from pathlib import Path
import json
import joblib
import sklearn
from .config import MODEL_PATH, PAYSIM_MODEL_PATH, BAF_MODEL_PATH


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
        if trained_version and trained_version != sklearn.__version__:
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
            preprocessing = joblib.load(path.with_name(stem + '.preprocessor.joblib'))
            model = XGBClassifier()
            model.load_model(path.with_name(stem + '.xgboost.json'))
            artifact = dict(report)
            artifact['pipeline'] = Pipeline([('preprocessing', preprocessing), ('model', model)])
        else:
            artifact = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f'Unable to load {expected_kind} model artifact ({type(exc).__name__})') from exc
    if (not isinstance(artifact, dict) or artifact.get('format_version') != (3 if portable else 2)
            or artifact.get('model_kind') != expected_kind or not artifact.get('features')
            or 'pipeline' not in artifact or 'threshold' not in artifact):
        raise RuntimeError(f'Unsupported {expected_kind} model artifact format')
    trained_version = artifact.get('library_versions', {}).get('scikit_learn')
    if trained_version and trained_version != sklearn.__version__:
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
            'warning': 'Результат модели на исследовательских данных, не подтверждённая вероятность мошенничества в банке.'}


def predict_paysim(features: dict):
    artifact = load_paysim_model()
    if artifact is None:
        return None
    expected = artifact['features']
    missing = [key for key in expected if key not in features]
    unknown = sorted(set(features) - set(expected))
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unsupported fields: {', '.join(unknown)}")
        raise ValueError('; '.join(detail))
    import pandas as pd
    frame = pd.DataFrame([{key: features[key] for key in expected}], columns=expected)
    frame['amount'] = pd.to_numeric(frame['amount'], errors='coerce')
    frame['oldbalanceOrg'] = pd.to_numeric(frame['oldbalanceOrg'], errors='coerce')
    frame['oldbalanceDest'] = pd.to_numeric(frame['oldbalanceDest'], errors='coerce')
    if frame[['amount', 'oldbalanceOrg', 'oldbalanceDest']].isna().any().any() or (frame[['amount', 'oldbalanceOrg', 'oldbalanceDest']] < 0).any().any():
        raise ValueError('amount, oldbalanceOrg and oldbalanceDest must be non-negative numbers')
    score = float(artifact['pipeline'].predict_proba(frame)[0, 1])
    return {'risk_score': round(score * 100, 2), 'threshold': artifact['threshold'],
            'flagged': score >= artifact['threshold'], 'source': 'paysim_ml',
            'model_version': artifact.get('model_version', 'unknown'),
            'score_kind': 'uncalibrated_ml_score',
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
            'warning': 'BAF Base is synthetic account-application data, not a calibrated fraud probability for a bank.'}
