"""Train a separate BAF Base account-application fraud detector.

This pipeline does not score transfers or payments. The last month is held
out for a single final evaluation; the preceding month is validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from experiment_tracking import finish_experiment, start_experiment
from train_ieee import file_provenance, metrics, operating_profiles, ranking_metrics, threshold_analysis

NUMERIC = [
    'income', 'name_email_similarity', 'prev_address_months_count',
    'current_address_months_count', 'customer_age', 'intended_balcon_amount',
    'proposed_credit_limit', 'session_length_in_minutes', 'bank_months_count',
    'email_is_free', 'phone_home_valid', 'phone_mobile_valid', 'has_other_cards',
    'foreign_request', 'keep_alive_session',
]
CATEGORICAL = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
FEATURES = NUMERIC + CATEGORICAL
REQUIRED = set(FEATURES) | {'fraud_bool', 'month'}
SENTINEL_MISSING = ['prev_address_months_count', 'current_address_months_count',
                    'bank_months_count', 'session_length_in_minutes']


def read_base(path: Path) -> pd.DataFrame:
    if path.name.lower() != 'base.csv':
        raise ValueError('Use only the BAF Base.csv variant; do not mix suite variants')
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = REQUIRED - columns
    if missing:
        raise ValueError(f'Missing BAF Base columns: {sorted(missing)}')
    dtypes = {name: 'float32' for name in NUMERIC}
    dtypes.update({name: 'category' for name in CATEGORICAL})
    dtypes.update({'fraud_bool': 'int8', 'month': 'int8'})
    data = pd.read_csv(path, usecols=sorted(REQUIRED), dtype=dtypes)
    if not data.fraud_bool.isin([0, 1]).all():
        raise ValueError('fraud_bool must be binary')
    if data.month.isna().any() or (data.month < 0).any():
        raise ValueError('month must be a nonnegative relative period')
    for name in SENTINEL_MISSING:
        data[name] = data[name].mask(data[name] == -1)
    return data


def split_months(data: pd.DataFrame):
    months = sorted(data.month.unique().tolist())
    if len(months) < 3:
        raise ValueError('BAF Base requires at least three months for train, validation, test')
    valid_month, test_month = months[-2:]
    parts = (data[data.month < valid_month], data[data.month == valid_month], data[data.month == test_month])
    for name, part in zip(('train', 'validation', 'test'), parts):
        if part.empty or part.fraud_bool.nunique() != 2:
            raise ValueError(f'{name} month split must contain both classes')
    return parts, {'train_months': months[:-2], 'validation_month': valid_month, 'test_month': test_month}


def train(data_path: Path, output: Path, n_estimators: int = 180, max_train_rows: int | None = None,
          mode: str = 'experiment', max_false_alerts_per_10000: float | None = 100,
          positive_weight_multiplier: float = 1.0):
    if output.exists() or output.with_suffix('.metrics.json').exists():
        raise FileExistsError('Output already exists; preserve earlier experiment artifacts')
    if n_estimators < 1 or mode not in {'experiment', 'final'}:
        raise ValueError('Use positive estimators and experiment or final mode')
    if not np.isfinite(positive_weight_multiplier) or positive_weight_multiplier <= 0:
        raise ValueError('positive_weight_multiplier must be a finite positive number')
    if mode == 'final' and positive_weight_multiplier != 1.0:
        raise ValueError('positive weight candidates are validation-only; final mode requires multiplier 1.0')
    if max_train_rows is not None and max_train_rows < 100:
        raise ValueError('max_train_rows must be at least 100')
    data = read_base(data_path)
    (train_rows, validation_rows, test_rows), boundaries = split_months(data)
    original_train_count = len(train_rows)
    if max_train_rows is not None:
        train_rows = train_rows.iloc[:max_train_rows]
    if train_rows.fraud_bool.nunique() != 2:
        raise ValueError('Selected training rows must contain both classes')
    experiment, started = start_experiment(
        dataset='Feedzai BAF Base', algorithm='XGBoost', features=FEATURES,
        split_method='month train/validation/final-test; final test omitted in experiment mode', mode=mode,
        hyperparameters={'n_estimators': n_estimators, 'max_train_rows': max_train_rows,
                         'max_false_alerts_per_10000': max_false_alerts_per_10000,
                         'max_depth': 4, 'learning_rate': .06, 'feature_set': 'base_v1',
                         'positive_weight_multiplier': positive_weight_multiplier},
    )
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True))])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20))])
    preprocessing = ColumnTransformer([('numeric', numeric, NUMERIC), ('categorical', categorical, CATEGORICAL)])
    y_train = train_rows.fraud_bool.astype('int8')
    pos = int(y_train.sum())
    model = XGBClassifier(n_estimators=n_estimators, max_depth=4, learning_rate=.06,
                          subsample=.85, colsample_bytree=.85, reg_lambda=4,
                          objective='binary:logistic', tree_method='hist', eval_metric='aucpr',
                          scale_pos_weight=((len(y_train) - pos) / pos) * positive_weight_multiplier,
                          random_state=42, n_jobs=min(4, os.cpu_count() or 1))
    pipeline = Pipeline([('preprocessing', preprocessing), ('model', model)])
    pipeline.fit(train_rows[FEATURES], y_train)
    baseline_pre = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scale', StandardScaler(with_mean=False))]), NUMERIC),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20))]), CATEGORICAL),
    ])
    baseline = Pipeline([('preprocessing', baseline_pre),
                         ('model', LogisticRegression(solver='liblinear', max_iter=500, tol=1e-3,
                                                      class_weight='balanced'))])
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=ConvergenceWarning)
        baseline.fit(train_rows[FEATURES], y_train)
    if baseline.named_steps['model'].n_iter_[0] >= 500:
        raise RuntimeError('BAF baseline failed to converge')
    valid_scores = pipeline.predict_proba(validation_rows[FEATURES])[:, 1]
    threshold, threshold_report = threshold_analysis(validation_rows.fraud_bool, valid_scores,
                                                    max_false_alerts_per_10000)
    valid_metrics = metrics(validation_rows.fraud_bool, valid_scores, threshold)
    encoded = preprocessing.get_feature_names_out()
    importance = sorted(zip(encoded, model.feature_importances_), key=lambda item: float(item[1]), reverse=True)
    report = {
        'format_version': 2, 'model_kind': 'baf_base', 'model_version': 'baf-base-xgboost-v1',
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'Feedzai BAF Base (synthetic account-opening applications)',
        'data_files': [file_provenance(data_path)],
        'library_versions': {'python': sys.version.split()[0], 'pandas': pd.__version__,
                             'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__},
        'features': FEATURES, 'numerical_features': NUMERIC, 'categorical_features': CATEGORICAL,
        'excluded_for_leakage': ['fraud_bool', 'month', 'device_fraud_count', 'days_since_request',
                                 'credit_risk_score', 'unverified historical aggregates'],
        'split_method': 'earlier months train; penultimate month validation; last month final test',
        'split_boundaries': boundaries,
        'split_counts': {'train_available': original_train_count, 'train_used': len(train_rows),
                         'validation': len(validation_rows), 'test': len(test_rows)},
        'experiment': finish_experiment(experiment, started),
        'threshold': float(threshold), 'threshold_analysis': threshold_report,
        'validation_ranking_metrics': ranking_metrics(validation_rows.fraud_bool, valid_scores),
        'validation_operating_profiles': operating_profiles(validation_rows.fraud_bool, valid_scores, threshold),
        'validation_metrics': valid_metrics, 'test_metrics': None, 'baseline_test_metrics': None,
        'baseline_convergence': {'solver': 'liblinear',
                                 'iterations': int(baseline.named_steps['model'].n_iter_[0]), 'max_iter': 500},
        'feature_importance': {'top_30': [{'feature': str(name), 'importance': round(float(value), 6)}
                                           for name, value in importance[:30]],
                               'leakage_audit': 'Only the explicit application-time whitelist is used; temporal availability still requires domain confirmation.'},
        'warning': 'Synthetic application data; model score is not a calibrated probability for a real bank.',
    }
    if mode == 'final':
        test_scores = pipeline.predict_proba(test_rows[FEATURES])[:, 1]
        report['test_metrics'] = metrics(test_rows.fraud_bool, test_scores, threshold)
        report['baseline_test_metrics'] = metrics(test_rows.fraud_bool,
                                                  baseline.predict_proba(test_rows[FEATURES])[:, 1], .5)
        artifact = dict(report)
        artifact['pipeline'] = pipeline
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'mode': mode, 'model_path': str(output) if mode == 'final' else None,
                      'validation': valid_metrics, 'test': report['test_metrics']}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='BAF Base.csv only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['experiment', 'final'], default='experiment')
    parser.add_argument('--estimators', type=int, default=180)
    parser.add_argument('--max-train-rows', type=int)
    parser.add_argument('--max-false-alerts-per-10000', type=float, default=100)
    parser.add_argument('--positive-weight-multiplier', type=float, default=1.0,
                        help='Experiment-only multiplier over inverse-prevalence class weight')
    args = parser.parse_args()
    train(args.data, args.output, args.estimators, args.max_train_rows, args.mode,
          args.max_false_alerts_per_10000, args.positive_weight_multiplier)


if __name__ == '__main__':
    main()
