"""Causal frequency features at inference: batch exactness + single-record fallback."""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import app.ml as ml
from app import ieee_freq


def _frame():
    return pd.DataFrame({'TransactionDT': [30, 10, 20, 40],
                         'card1': ['A', 'A', 'B', 'A'],
                         'P_emaildomain': ['x', 'x', 'y', 'x']})


def test_batch_mode_exact_causal_counts_log_scale():
    out = ieee_freq.add_freq_features(_frame(), {'freq_sources': ieee_freq.FREQ_SOURCES,
                                                 'freq_scale': 'log'}, scale='log')
    # Rows are sorted by TransactionDT ascending: times 10, 20, 30, 40.
    assert out['TransactionDT'].tolist() == [10, 20, 30, 40]
    assert np.allclose(out['card1_freq'].tolist(), np.log1p([0, 0, 1, 2]))
    assert np.allclose(out['p_email_freq'].tolist(), np.log1p([0, 0, 1, 2]))


def test_single_record_uses_freq_state():
    artifact = {'freq_sources': ieee_freq.FREQ_SOURCES,
                'freq_state': {'card1_freq': {'A': 7}, 'p_email_freq': {}}}
    row = pd.DataFrame({'card1': ['A'], 'P_emaildomain': ['unknown-domain']})
    out = ieee_freq.add_freq_features(row, artifact, scale='log')
    assert np.isclose(out['card1_freq'].iloc[0], np.log1p(7))
    assert out['p_email_freq'].iloc[0] == 0.0
    # Missing key in state maps to 0, not NaN.
    assert out['device_freq'].iloc[0] == 0.0 if 'device_freq' in out else True


def test_train_freq_state_counts():
    state = ieee_freq.train_freq_state(pd.DataFrame({'card1': ['A', 'B', 'A', 'A']}))
    assert state['card1_freq'] == {'A': 3, 'B': 1}


def test_predict_ieee_with_freq_artifact(monkeypatch, tmp_path):
    nums = ['TransactionAmt', 'card1', 'card1_freq']
    cats = ['ProductCD']
    features = nums + cats
    rng = np.random.RandomState(0)
    frame = pd.DataFrame({'TransactionAmt': rng.gamma(20.0, 10.0, 100),
                          'card1': rng.choice([101, 202], 100).astype('float32'),
                          'card1_freq': np.log1p(rng.randint(0, 50, 100)),
                          'ProductCD': rng.choice(['W', 'C'], 100)})
    y = (frame['card1_freq'] > frame['card1_freq'].median()).astype(int)
    transform = ColumnTransformer([
        ('numeric', 'passthrough', nums),
        ('category', OneHotEncoder(handle_unknown='ignore'), cats)])
    model = LogisticRegression(max_iter=200).fit(transform.fit_transform(frame), y)
    artifact = {'format_version': 1, 'model_version': 'test-freq-ieee',
                'pipeline': Pipeline([('preprocessing', transform), ('model', model)]),
                'features': features, 'numerical_features': nums, 'categorical_features': cats,
                'threshold': 0.5, 'freq_sources': {'card1_freq': ('card1',)},
                'freq_scale': 'log', 'freq_state': {'card1_freq': {'101': 3}}}
    monkeypatch.setattr(ml, 'load_model', lambda: artifact)
    result = ml.predict_ieee({'TransactionAmt': 250.0, 'card1': 101, 'ProductCD': 'W'})
    assert 0.0 <= result['risk_score'] <= 100.0
    assert result['source'] == 'ieee_cis_ml'
