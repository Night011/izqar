"""Independently trained IEEE-CIS model. No synthetic transfers fed to IEEE model."""
from functools import lru_cache
from pathlib import Path
import joblib
from .config import MODEL_PATH, PAYSIM_MODEL_PATH

@lru_cache(maxsize=1)
def load_model():
    if not Path(MODEL_PATH).is_file():
        return None
    artifact = joblib.load(MODEL_PATH)
    if artifact.get('format_version') != 2 or artifact.get('model_kind') != 'ieee_cis' or not artifact.get('features'):
        raise RuntimeError('Unsupported model artifact format')
    return artifact


@lru_cache(maxsize=1)
def load_paysim_model():
    if not PAYSIM_MODEL_PATH.is_file():
        return None
    artifact = joblib.load(PAYSIM_MODEL_PATH)
    if artifact.get('format_version') != 2 or artifact.get('model_kind') != 'paysim' or not artifact.get('features'):
        raise RuntimeError('Unsupported PaySim model artifact format')
    return artifact


def predict_ieee(features: dict):
    artifact = load_model()
    if artifact is None:
        return None
    expected = artifact['features']
    supplied = set(features)
    # Prevent a confident answer on a row with no meaningful model inputs.
    recognized = supplied.intersection(expected)
    if len(recognized) < 3 or 'TransactionAmt' not in recognized:
        raise ValueError('Provide TransactionAmt and at least two supported IEEE-CIS features')
    record = {key: features.get(key) for key in expected}
    import pandas as pd
    frame = pd.DataFrame([record], columns=expected)
    for col in artifact['numerical_features']:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
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
