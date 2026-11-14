"""Local ML explanations must be bounded and avoid echoing feature values."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.ml import _local_xgboost_explanation


def test_local_xgboost_explanation_returns_named_log_odds_contributions_only():
    rows = pd.DataFrame({'amount': [1.0, 2.0, 7.0, 9.0], 'balance': [9.0, 8.0, 3.0, 1.0]})
    preprocessing = ColumnTransformer([('numeric', 'passthrough', ['amount', 'balance'])])
    pipeline = Pipeline([('preprocessing', preprocessing),
                         ('model', XGBClassifier(n_estimators=3, max_depth=2, learning_rate=.3,
                                                 n_jobs=1, random_state=7, eval_metric='logloss'))])
    pipeline.fit(rows, [0, 0, 1, 1])

    explanation = _local_xgboost_explanation({'pipeline': pipeline}, rows.iloc[[3]])

    assert explanation
    assert len(explanation) <= 5
    assert {item['feature'] for item in explanation} <= {'numeric__amount', 'numeric__balance'}
    assert all(set(item) == {'feature', 'log_odds_contribution', 'direction'} for item in explanation)
    assert all(item['direction'] in {'increases_model_score', 'decreases_model_score'} for item in explanation)
