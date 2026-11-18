"""Error analysis for the v1 BAF configuration on month-5 validation.

Trains v1 on months [0..4] (no artifact persisted), evaluates on month 5,
then reports: metrics, mean |SHAP| (via XGBoost pred_contribs), and
segment-level FP/FN tables. Read-only w.r.t. months 6/7.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, DMatrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baf import NUMERIC, CATEGORICAL, FEATURES, read_base, split_months  # noqa: E402
from train_ieee import metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--sample', type=int, default=20000)
    parser.add_argument('--output', type=Path, default=Path('models/experiments/baf_error_analysis.json'))
    args = parser.parse_args()

    data = read_base(args.data)
    parts, boundaries = split_months(data)
    train_rows = parts[0][parts[0].month <= 4]
    valid_rows = parts[0][parts[0].month == 5]
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True))])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20))])
    preprocessing = ColumnTransformer([('numeric', numeric, NUMERIC),
                                      ('categorical', categorical, CATEGORICAL)])
    y = train_rows.fraud_bool.astype('int8')
    pos = int(y.sum())
    model = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=.06,
                          subsample=.85, colsample_bytree=.85, reg_lambda=4,
                          objective='binary:logistic', tree_method='hist', eval_metric='aucpr',
                          scale_pos_weight=(len(y) - pos) / pos,
                          random_state=42, n_jobs=4, early_stopping_rounds=40)
    x_tr = preprocessing.fit_transform(train_rows[FEATURES])
    x_va = preprocessing.transform(valid_rows[FEATURES])
    model.fit(x_tr, y, eval_set=[(x_va, valid_rows.fraud_bool.astype('int8'))], verbose=False)
    y_va = valid_rows.fraud_bool.to_numpy()
    scores = model.predict_proba(x_va)[:, 1]
    threshold = float(np.quantile(scores, 0.99))
    print(json.dumps(metrics(valid_rows.fraud_bool, scores, threshold), indent=2))

    names = list(preprocessing.get_feature_names_out())
    imp = sorted(zip(names, model.feature_importances_), key=lambda kv: -kv[1])
    print('\nTop 20 importances:')
    for name, value in imp[:20]:
        print(f'  {name:45s} {value:.4f}')

    idx = np.random.RandomState(42).choice(x_va.shape[0], min(args.sample, x_va.shape[0]), replace=False)
    contrib = model.get_booster().predict(DMatrix(x_va[idx], feature_names=names), pred_contribs=True)[:, :-1]
    mean_abs = np.abs(contrib).mean(axis=0)
    shap_rank = sorted(zip(names, mean_abs), key=lambda kv: -kv[1])
    print('\nTop 20 mean |SHAP|:')
    for name, value in shap_rank[:20]:
        print(f'  {name:45s} {value:.5f}')

    err = valid_rows.copy()
    err['score'] = scores
    err['fp'] = ((scores >= threshold) & (y_va == 0)).astype(int)
    err['fn'] = ((scores < threshold) & (y_va == 1)).astype(int)
    report = {'threshold': threshold, 'validation_month': 5, 'segments': {}}
    for seg in ['employment_status', 'housing_status', 'device_os', 'source',
                'email_is_free', 'payment_type']:
        grouped = err.groupby(seg, dropna=False).agg(rows=('fp', 'size'), fp=('fp', 'sum'),
                                                     fn=('fn', 'sum'), frauds=('fraud_bool', 'sum'))
        grouped['fp_rate'] = grouped.fp / (grouped.rows - grouped.frauds).clip(lower=1)
        grouped['fn_share_of_frauds'] = grouped.fn / grouped.frauds.clip(lower=1)
        report['segments'][seg] = grouped.reset_index().to_dict(orient='records')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(f'\nsegment report -> {args.output}')


if __name__ == '__main__':
    main()
