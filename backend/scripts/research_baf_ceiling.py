"""Is the BAF ceiling set by drift or by the available signal?

For each research month the model is fitted on one random half of that month and
scored on the other half. That removes month-to-month drift and keeps only the
within-period signal, so the gap to the cross-month numbers (0.150/0.175/0.187
for v1v) is the cost of drift rather than a tuning deficit.

Half-split only, fixed 300 rounds, no early stopping and no threshold selection:
PR-AUC and P@1000 are threshold-free, and the labels of the held-out half never
touch the fit. Months 6/7 are never read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baf import (CATEGORICAL, FEATURE_SETS, VELOCITY, add_engineered, read_base,  # noqa: E402
                       split_months)
from train_ieee import ranking_metrics  # noqa: E402

ROUNDS = 300


def fit_and_score(rows_a, rows_b, feature_cols, numeric_cols, seed):
    y_a = rows_a.fraud_bool.astype('int8')
    positive = int(y_a.sum())
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True))]),
         numeric_cols),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20))]),
         CATEGORICAL)])
    model = XGBClassifier(n_estimators=ROUNDS, max_depth=4, learning_rate=.06, subsample=.85,
                          colsample_bytree=.85, reg_lambda=4.0,
                          scale_pos_weight=(len(y_a) - positive) / positive,
                          objective='binary:logistic', tree_method='hist', eval_metric='aucpr',
                          random_state=seed, n_jobs=min(4, os.cpu_count() or 1))
    model.fit(preprocessing.fit_transform(rows_a[feature_cols]), y_a, verbose=False)
    scores = model.predict_proba(preprocessing.transform(rows_b[feature_cols]))[:, 1]
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('models/experiments/baf_ceiling.json'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-curve', action='store_true',
                        help='also score month 4 with the train region subsampled, to expose the data slope')
    args = parser.parse_args()

    data = add_engineered(read_base(args.data, VELOCITY))
    train_rows = split_months(data)[0][0]
    rng = np.random.default_rng(args.seed)
    report = {'seed': args.seed, 'rounds': ROUNDS, 'months': {}}
    for month in (3, 4, 5):
        rows = train_rows[train_rows.month == month]
        order = rng.permutation(len(rows))
        rows_a = rows.iloc[order[:len(rows) // 2]]
        rows_b = rows.iloc[order[len(rows) // 2:]]
        y_b = rows_b.fraud_bool.to_numpy()
        entry = {'rows_a': len(rows_a), 'rows_b': len(rows_b),
                 'fraud_rate_a': round(float(rows_a.fraud_bool.mean()), 6),
                 'fraud_rate_b': round(float(rows_b.fraud_bool.mean()), 6)}
        for name in ('v1', 'v1v'):
            numeric_cols, feature_cols = FEATURE_SETS[name]
            scores = fit_and_score(rows_a, rows_b, feature_cols, numeric_cols, args.seed)
            ranking = ranking_metrics(y_b, scores)
            entry[name] = {'pr_auc': round(float(average_precision_score(y_b, scores)), 6),
                           'precision_at_100': ranking['100']['precision_at_k'],
                           'precision_at_1000': ranking['1000']['precision_at_k']}
            print(f"month {month} {name}: same-month PR-AUC {entry[name]['pr_auc']:.6f} "
                  f"P@1000 {entry[name].get('precision_at_1000')}", flush=True)
        report['months'][str(month)] = entry

    if args.learning_curve:
        external = train_rows[train_rows.month == 4]
        y_external = external.fraud_bool.to_numpy()
        pool = train_rows[train_rows.month < 3]
        order = rng.permutation(len(pool))
        numeric_cols, feature_cols = FEATURE_SETS['v1v']
        report['learning_curve'] = {'external_month': 4, 'pool_rows': int(len(pool)), 'points': []}
        for fraction in (0.25, 0.5, 1.0):
            subset = pool.iloc[order[:max(int(fraction * len(pool)), 2)]]
            scores = fit_and_score(subset, external, feature_cols, numeric_cols, args.seed)
            point = {'fraction': fraction, 'rows': int(len(subset)),
                     'pr_auc': round(float(average_precision_score(y_external, scores)), 6)}
            report['learning_curve']['points'].append(point)
            print(f"learning curve {fraction:.2f} ({point['rows']} rows): "
                  f"PR-AUC {point['pr_auc']:.6f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote', args.output)


if __name__ == '__main__':
    main()
