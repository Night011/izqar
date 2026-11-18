"""Inference contract tests for the IEEE-CIS artifact path (app.ml)."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import app.ml as ml


def _synthetic_artifact(tmp_path):
    nums = ['TransactionAmt', 'card1']
    cats = ['ProductCD']
    features = nums + cats
    rng = np.random.RandomState(0)
    frame = pd.DataFrame({
        'TransactionAmt': rng.gamma(20.0, 10.0, 200),
        'card1': rng.randint(1000, 20000, 200).astype('float32'),
        'ProductCD': rng.choice(['W', 'C', 'R'], 200),
    })
    y = (frame['TransactionAmt'] > frame['TransactionAmt'].median()).astype(int)
    transform = ColumnTransformer([
        ('numeric', 'passthrough', nums),
        ('category', OneHotEncoder(handle_unknown='ignore'), cats),
    ])
    model = LogisticRegression(max_iter=200).fit(transform.fit_transform(frame), y)
    pipeline = Pipeline([('preprocessing', transform), ('model', model)])
    artifact = {'format_version': 1, 'model_version': 'test-ieee', 'pipeline': pipeline,
                'features': features, 'numerical_features': nums,
                'categorical_features': cats, 'threshold': 0.5}
    path = tmp_path / 'ieee_test.joblib'
    joblib.dump(artifact, path)
    return artifact


def test_predict_ieee_roundtrip(tmp_path, monkeypatch):
    artifact = _synthetic_artifact(tmp_path)
    monkeypatch.setattr(ml, 'load_model', lambda: artifact)
    result = ml.predict_ieee({'TransactionAmt': 250.0, 'card1': 1234, 'ProductCD': 'W'})
    assert result['source'] == 'ieee_cis_ml'
    assert 0.0 <= result['risk_score'] <= 100.0
    assert isinstance(result['flagged'], bool)
    assert result['threshold'] == 0.5


def test_predict_ieee_rejects_empty_features(monkeypatch, tmp_path):
    artifact = _synthetic_artifact(tmp_path)
    monkeypatch.setattr(ml, 'load_model', lambda: artifact)
    with pytest.raises(ValueError):
        ml.predict_ieee({'TransactionAmt': 10.0})  # fewer than 3 recognized features


def test_predict_ieee_returns_none_without_model(monkeypatch):
    monkeypatch.setattr(ml, 'load_model', lambda: None)
    assert ml.predict_ieee({'TransactionAmt': 10.0, 'card1': 1, 'ProductCD': 'W'}) is None
