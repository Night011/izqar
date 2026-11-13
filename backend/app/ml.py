"""Independently trained IEEE-CIS model. No synthetic transfers fed to IEEE model."""
from functools import lru_cache
from pathlib import Path
import joblib
from .config import MODEL_PATH

@lru_cache(maxsize=1)
def load_model():
    if not Path(MODEL_PATH).is_file():
        return None
    artifact = joblib.load(MODEL_PATH)
    if artifact.get('format_version') != 1 or not artifact.get('features'):
        raise RuntimeError('Unsupported model artifact format')
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
            'source': 'ieee_cis_ml',
            'warning': 'Результат модели на исследовательских данных, не подтверждённая вероятность мошенничества в банке.'}
