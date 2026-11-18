"""Research-only ablation: v2 features + velocity/aggregate columns.

These columns (zip_count_4w, velocity_6h/24h/4w, bank_branch_count_8w,
date_of_birth_distinct_emails_4w, device_distinct_emails_8w) are behavioral
aggregates whose temporal availability at *application* time is not verified
for IZQAR's real-time inference contract. They are evaluated ONLY here as a
headroom probe — never shipped in the deployable candidate without a verified
streaming aggregation pipeline.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baf import CATEGORICAL, NUMERIC_V2, add_engineered, read_base, split_months  # noqa: E402
from train_ieee import metrics, ranking_metrics, threshold_analysis  # noqa: E402

VELOCITY = ['zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w',
            'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 'device_distinct_emails_8w']
NUMERIC_V2V = NUMERIC_V2 + VELOCITY
FEATURES_V2V = NUMERIC_V2V + CATEGORICAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--estimators', type=int, default=400)
    parser.add_argument('--early-stopping-rounds', type=int, default=40)
    args = parser.parse_args()

    data = read_base(args.data)
    raw = pd.read_csv(args.data, usecols=VELOCITY, dtype='float32')
    data = pd.concat([data.reset_index(drop=True), raw.reset_index(drop=True)], axis=1)
    data = add_engineered(data)
    parts, boundaries = split_months(data)
    train_rows = parts[0]
    months = sorted(train_rows.month.unique().tolist())
    folds = []
    for k in range(3, len(months)):
        fold_train = train_rows[train_rows.month.isin(months[:k])]
        fold_valid = train_rows[train_rows.month == months[k]]
        numeric = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True))])
        categorical = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20))])
        preprocessing = ColumnTransformer([('numeric', numeric, NUMERIC_V2V),
                                          ('categorical', categorical, CATEGORICAL)])
        y_fold = fold_train.fraud_bool.astype('int8')
        pos = int(y_fold.sum())
        model = XGBClassifier(n_estimators=args.estimators, max_depth=4, learning_rate=.06,
                              subsample=.85, colsample_bytree=.85, reg_lambda=4,
                              objective='binary:logistic', tree_method='hist', eval_metric='aucpr',
                              scale_pos_weight=(len(y_fold) - pos) / pos,
                              random_state=42, n_jobs=min(4, os.cpu_count() or 1),
                              early_stopping_rounds=args.early_stopping_rounds)
        x_tr = preprocessing.fit_transform(fold_train[FEATURES_V2V])
        x_va = preprocessing.transform(fold_valid[FEATURES_V2V])
        model.fit(x_tr, y_fold, eval_set=[(x_va, fold_valid.fraud_bool.astype('int8'))], verbose=False)
        scores = model.predict_proba(x_va)[:, 1]
        threshold, _ = threshold_analysis(fold_valid.fraud_bool, scores, 100)
        fold_metrics = metrics(fold_valid.fraud_bool, scores, threshold)
        fold_metrics.update(ranking_metrics(fold_valid.fraud_bool, scores))
        folds.append({'valid_month': months[k], 'metrics': fold_metrics,
                      'best_iteration': getattr(model, 'best_iteration', None)})
        print(f"velocity fold valid month {months[k]}: PR-AUC {fold_metrics['average_precision_pr_auc']:.6f} "
              f"F1 {fold_metrics['f1']:.6f}", flush=True)
    mean_metrics = {key: round(float(np.mean([f['metrics'][key] for f in folds])), 6)
                    for key in folds[0]['metrics'] if isinstance(folds[0]['metrics'][key], (int, float))}
    report = {'mode': 'research-velocity-ablation', 'trained_at_utc': datetime.now(timezone.utc).isoformat(),
              'features': FEATURES_V2V, 'folds': folds, 'mean_metrics': mean_metrics,
              'note': 'Headroom probe only. Velocity/count aggregates are NOT eligible for IZQAR real-time inference without a verified streaming pipeline.'}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(mean_metrics, indent=2))


if __name__ == '__main__':
    main()
