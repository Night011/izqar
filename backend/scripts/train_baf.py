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
from train_ieee import (calibration_diagnostics, file_provenance, metrics, operating_profiles, ranking_metrics,
                        temporal_drift_diagnostics, threshold_analysis)

NUMERIC = [
    'income', 'name_email_similarity', 'prev_address_months_count',
    'current_address_months_count', 'customer_age', 'intended_balcon_amount',
    'proposed_credit_limit', 'session_length_in_minutes', 'bank_months_count',
    'email_is_free', 'phone_home_valid', 'phone_mobile_valid', 'has_other_cards',
    'foreign_request', 'keep_alive_session',
]
ENGINEERED_NUMERIC = ['income_log', 'credit_limit_log', 'limit_to_income', 'address_months_total',
                      'address_churn_prev_share', 'tenure_per_age', 'session_share_limit',
                      'young_client', 'email_free_x_foreign', 'phones_valid_x2',
                      'limit_to_income_log', 'age_income_ratio']
NUMERIC_V2 = NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
FEATURES = NUMERIC + CATEGORICAL
FEATURES_V2 = NUMERIC_V2 + CATEGORICAL
FEATURE_SETS = {'v1': (NUMERIC, FEATURES), 'v2': (NUMERIC_V2, FEATURES_V2)}
REQUIRED = set(FEATURES) | {'fraud_bool', 'month'}
SENTINEL_MISSING = ['prev_address_months_count', 'current_address_months_count',
                    'bank_months_count', 'session_length_in_minutes']


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num = lambda c: pd.to_numeric(df[c], errors='coerce')
    income = num('income').clip(lower=0)
    limit = num('proposed_credit_limit').clip(lower=0)
    prev = num('prev_address_months_count')
    current = num('current_address_months_count')
    df['income_log'] = np.log1p(income)
    df['credit_limit_log'] = np.log1p(limit)
    df['limit_to_income'] = (limit / (income + 1.0)).clip(upper=1e6)
    df['limit_to_income_log'] = np.log1p(df['limit_to_income'])
    df['address_months_total'] = prev.fillna(0) + current.fillna(0)
    df['address_churn_prev_share'] = prev / (prev + current + 1.0)
    df['tenure_per_age'] = num('bank_months_count') / (num('customer_age') + 1.0)
    df['session_share_limit'] = num('session_length_in_minutes') / limit.clip(lower=1.0)
    df['young_client'] = (num('customer_age') < 25).astype('float32')
    df['email_free_x_foreign'] = num('email_is_free') * num('foreign_request')
    df['phones_valid_x2'] = num('phone_home_valid') * num('phone_mobile_valid')
    df['age_income_ratio'] = num('customer_age') / (income + 1.0)
    return df


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


def run_research(train_rows: pd.DataFrame, boundaries: dict, n_estimators: int,
                 max_false_alerts_per_10000: float | None, positive_weight_multiplier: float,
                 early_stopping_rounds: int | None, feature_set: str, output: Path, max_depth: int = 4):
    """Expanding month folds over the *train region only*. Penultimate month (validation)
    and the last month (final test) are never touched; no artifact is written."""
    numeric_cols, feature_cols = FEATURE_SETS[feature_set]
    months = boundaries['train_months']
    fold_reports = []
    mean_keys = None
    for k in range(3, len(months)):
        fold_train = train_rows[train_rows.month.isin(months[:k])]
        fold_valid = train_rows[train_rows.month == months[k]]
        if fold_valid.empty or fold_valid.fraud_bool.nunique() != 2 or fold_train.fraud_bool.nunique() != 2:
            continue
        numeric = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True))])
        categorical = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20))])
        preprocessing = ColumnTransformer([('numeric', numeric, numeric_cols),
                                          ('categorical', categorical, CATEGORICAL)])
        y_fold = fold_train.fraud_bool.astype('int8')
        pos = int(y_fold.sum())
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=.06,
                              subsample=.85, colsample_bytree=.85, reg_lambda=4,
                              objective='binary:logistic', tree_method='hist', eval_metric='aucpr',
                              scale_pos_weight=((len(y_fold) - pos) / pos) * positive_weight_multiplier,
                              random_state=42, n_jobs=min(4, os.cpu_count() or 1),
                              early_stopping_rounds=early_stopping_rounds)
        x_tr = preprocessing.fit_transform(fold_train[feature_cols])
        x_va = preprocessing.transform(fold_valid[feature_cols])
        model.fit(x_tr, y_fold, eval_set=[(x_va, fold_valid.fraud_bool.astype('int8'))], verbose=False)
        scores = model.predict_proba(x_va)[:, 1]
        threshold, threshold_report = threshold_analysis(fold_valid.fraud_bool, scores,
                                                         max_false_alerts_per_10000)
        fold_metrics = metrics(fold_valid.fraud_bool, scores, threshold)
        fold_metrics.update(ranking_metrics(fold_valid.fraud_bool, scores))
        fold_report = {'train_months': months[:k], 'valid_month': months[k],
                       'train_rows': len(fold_train), 'valid_rows': len(fold_valid),
                       'best_iteration': getattr(model, 'best_iteration', None),
                       'threshold': round(float(threshold), 6), 'metrics': fold_metrics}
        fold_reports.append(fold_report)
        mean_keys = list(fold_metrics)
        print(f"fold train months {months[:k]} valid month {months[k]}: "
              f"PR-AUC {fold_metrics['average_precision_pr_auc']:.6f} F1 {fold_metrics['f1']:.6f} "
              f"best_iter {fold_report['best_iteration']}", flush=True)
    mean_metrics = {key: round(float(np.mean([r['metrics'][key] for r in fold_reports])), 6)
                    for key in (mean_keys or [])
                    if isinstance(fold_reports[0]['metrics'][key], (int, float))}
    report = {'mode': 'research', 'feature_set': feature_set,
              'trained_at_utc': datetime.now(timezone.utc).isoformat(),
              'n_estimators': n_estimators, 'positive_weight_multiplier': positive_weight_multiplier,
              'early_stopping_rounds': early_stopping_rounds,
              'max_false_alerts_per_10000': max_false_alerts_per_10000,
              'split_method': 'expanding month folds within the train region; penultimate/final months untouched',
              'excluded_for_leakage': ['fraud_bool', 'month', 'device_fraud_count', 'days_since_request',
                                       'credit_risk_score', 'unverified historical aggregates'],
              'folds': fold_reports, 'mean_metrics': mean_metrics,
              'features': feature_cols}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'mean_metrics': mean_metrics}, indent=2))
    return report


def train(data_path: Path, output: Path, n_estimators: int = 180, max_train_rows: int | None = None,
          mode: str = 'experiment', max_false_alerts_per_10000: float | None = 100,
          positive_weight_multiplier: float = 1.0, early_stopping_rounds: int | None = None,
          feature_set: str = 'v1', max_depth: int = 4):
    if mode not in {'experiment', 'final', 'research'}:
        raise ValueError('Use experiment, final or research mode')
    if feature_set not in FEATURE_SETS:
        raise ValueError(f'feature_set must be one of {sorted(FEATURE_SETS)}')
    if mode != 'research' and (output.exists() or output.with_suffix('.metrics.json').exists()):
        raise FileExistsError('Output already exists; preserve earlier experiment artifacts')
    if n_estimators < 1:
        raise ValueError('Use positive estimators')
    if not np.isfinite(positive_weight_multiplier) or positive_weight_multiplier <= 0:
        raise ValueError('positive_weight_multiplier must be a finite positive number')
    if mode == 'final' and positive_weight_multiplier != 1.0:
        raise ValueError('positive weight candidates are validation-only; final mode requires multiplier 1.0')
    if early_stopping_rounds is not None and early_stopping_rounds < 1:
        raise ValueError('early_stopping_rounds must be positive when supplied')
    if max_train_rows is not None and max_train_rows < 100:
        raise ValueError('max_train_rows must be at least 100')
    data = read_base(data_path)
    numeric_cols, feature_cols = FEATURE_SETS[feature_set]
    if feature_set == 'v2':
        data = add_engineered(data)
    (train_rows, validation_rows, test_rows), boundaries = split_months(data)
    if mode == 'research':
        return run_research(train_rows, boundaries, n_estimators, max_false_alerts_per_10000,
                            positive_weight_multiplier, early_stopping_rounds, feature_set, output, max_depth)
    original_train_count = len(train_rows)
    if max_train_rows is not None:
        train_rows = train_rows.iloc[:max_train_rows]
    if train_rows.fraud_bool.nunique() != 2:
        raise ValueError('Selected training rows must contain both classes')
    experiment, started = start_experiment(
        dataset='Feedzai BAF Base', algorithm='XGBoost', features=feature_cols,
        split_method='month train/validation/final-test; final test omitted in experiment mode', mode=mode,
        hyperparameters={'n_estimators': n_estimators, 'max_train_rows': max_train_rows,
                         'max_false_alerts_per_10000': max_false_alerts_per_10000,
                         'max_depth': max_depth, 'learning_rate': .06, 'feature_set': feature_set,
                         'positive_weight_multiplier': positive_weight_multiplier,
                         'early_stopping_rounds': early_stopping_rounds},
    )
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True))])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20))])
    preprocessing = ColumnTransformer([('numeric', numeric, numeric_cols), ('categorical', categorical, CATEGORICAL)])
    y_train = train_rows.fraud_bool.astype('int8')
    pos = int(y_train.sum())
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=.06,
                          subsample=.85, colsample_bytree=.85, reg_lambda=4,
                          objective='binary:logistic', tree_method='hist', eval_metric='aucpr',
                          scale_pos_weight=((len(y_train) - pos) / pos) * positive_weight_multiplier,
                          random_state=42, n_jobs=min(4, os.cpu_count() or 1),
                          early_stopping_rounds=early_stopping_rounds)
    pipeline = Pipeline([('preprocessing', preprocessing), ('model', model)])
    # Fit transforms on earlier months only.  The penultimate month is used
    # only for optional early stopping and operating-point selection.
    x_train_transformed = preprocessing.fit_transform(train_rows[feature_cols])
    x_valid_transformed = preprocessing.transform(validation_rows[feature_cols])
    model.fit(x_train_transformed, y_train,
              eval_set=[(x_valid_transformed, validation_rows.fraud_bool.astype('int8'))], verbose=False)
    baseline_pre = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scale', StandardScaler(with_mean=False))]), numeric_cols),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20))]), CATEGORICAL),
    ])
    baseline = Pipeline([('preprocessing', baseline_pre),
                         ('model', LogisticRegression(solver='liblinear', max_iter=500, tol=1e-3,
                                                      class_weight='balanced'))])
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=ConvergenceWarning)
        baseline.fit(train_rows[feature_cols], y_train)
    if baseline.named_steps['model'].n_iter_[0] >= 500:
        raise RuntimeError('BAF baseline failed to converge')
    valid_scores = model.predict_proba(x_valid_transformed)[:, 1]
    threshold, threshold_report = threshold_analysis(validation_rows.fraud_bool, valid_scores,
                                                    max_false_alerts_per_10000)
    valid_metrics = metrics(validation_rows.fraud_bool, valid_scores, threshold)
    validation_drift = temporal_drift_diagnostics(train_rows[feature_cols], validation_rows[feature_cols], numeric_cols, CATEGORICAL)
    encoded = preprocessing.get_feature_names_out()
    importance = sorted(zip(encoded, model.feature_importances_), key=lambda item: float(item[1]), reverse=True)
    report = {
        'format_version': 2, 'model_kind': 'baf_base', 'model_version': 'baf-base-xgboost-v1', 'run_mode': mode,
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'Feedzai BAF Base (synthetic account-opening applications)',
        'data_files': [file_provenance(data_path)],
        'library_versions': {'python': sys.version.split()[0], 'pandas': pd.__version__,
                             'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__},
        'features': feature_cols, 'numerical_features': numeric_cols, 'categorical_features': CATEGORICAL,
        'excluded_for_leakage': ['fraud_bool', 'month', 'device_fraud_count', 'days_since_request',
                                 'credit_risk_score', 'unverified historical aggregates'],
        'split_method': 'earlier months train; penultimate month validation; last month final test',
        'split_boundaries': boundaries,
        'split_counts': {'train_available': original_train_count, 'train_used': len(train_rows),
                         'validation': len(validation_rows), 'test': len(test_rows)},
        'experiment': finish_experiment(experiment, started),
        'threshold': float(threshold), 'threshold_analysis': threshold_report,
        'validation_calibration': calibration_diagnostics(validation_rows.fraud_bool, valid_scores),
        'validation_temporal_drift': validation_drift,
        'validation_ranking_metrics': ranking_metrics(validation_rows.fraud_bool, valid_scores),
        'validation_operating_profiles': operating_profiles(validation_rows.fraud_bool, valid_scores, threshold),
        'validation_metrics': valid_metrics, 'test_metrics': None, 'baseline_test_metrics': None,
        'baseline_convergence': {'solver': 'liblinear',
                                 'iterations': int(baseline.named_steps['model'].n_iter_[0]), 'max_iter': 500},
        'xgboost_training': {'early_stopping_rounds': early_stopping_rounds,
                             'best_iteration': getattr(model, 'best_iteration', None),
                             'selection_data': 'validation month only; final test is never passed to fit'},
        'feature_importance': {'top_30': [{'feature': str(name), 'importance': round(float(value), 6)}
                                           for name, value in importance[:30]],
                               'leakage_audit': 'Only the explicit application-time whitelist is used; temporal availability still requires domain confirmation.'},
        'warning': 'Synthetic application data; model score is not a calibrated probability for a real bank.',
    }
    if mode == 'final':
        test_scores = pipeline.predict_proba(test_rows[feature_cols])[:, 1]
        report['test_metrics'] = metrics(test_rows.fraud_bool, test_scores, threshold)
        report['baseline_test_metrics'] = metrics(test_rows.fraud_bool,
                                                  baseline.predict_proba(test_rows[feature_cols])[:, 1], .5)
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
    parser.add_argument('--mode', choices=['experiment', 'final', 'research'], default='experiment')
    parser.add_argument('--feature-set', choices=['v1', 'v2'], default='v1')
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--estimators', type=int, default=180)
    parser.add_argument('--max-train-rows', type=int)
    parser.add_argument('--max-false-alerts-per-10000', type=float, default=100)
    parser.add_argument('--positive-weight-multiplier', type=float, default=1.0,
                        help='Experiment/research-only multiplier over inverse-prevalence class weight')
    parser.add_argument('--early-stopping-rounds', type=int,
                        help='Stop after this many non-improving validation PR-AUC rounds')
    args = parser.parse_args()
    train(args.data, args.output, args.estimators, args.max_train_rows, args.mode,
          args.max_false_alerts_per_10000, args.positive_weight_multiplier,
          args.early_stopping_rounds, args.feature_set, args.max_depth)


if __name__ == '__main__':
    main()
