"""Research-only algorithm comparison for BAF: XGBoost vs LightGBM vs CatBoost.

Identical expanding month folds, identical hyperparameters family, identical
class weighting (scale_pos_weight), v1 and v2 feature sets. No artifact is
written; the penultimate and final months are never touched.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baf import CATEGORICAL, FEATURE_SETS, add_engineered, read_base, split_months  # noqa: E402
from train_ieee import metrics, ranking_metrics, threshold_analysis  # noqa: E402


def make_preprocess(numeric_cols):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True))])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20))])
    return ColumnTransformer([('numeric', numeric, numeric_cols),
                              ('categorical', categorical, CATEGORICAL)])


def make_model(algo, spw, seed, device='cuda'):
    if algo == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=.06,
                             subsample=.85, colsample_bytree=.85, reg_lambda=4,
                             objective='binary:logistic', tree_method='hist', eval_metric='aucpr',
                             scale_pos_weight=spw, random_state=seed, n_jobs=min(4, os.cpu_count() or 1),
                             early_stopping_rounds=40, device=device)
    if algo == 'lightgbm':
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        return LGBMClassifier(n_estimators=400, max_depth=4, num_leaves=15, learning_rate=.06,
                              subsample=.85, colsample_bytree=.85, reg_lambda=4,
                              scale_pos_weight=spw, random_state=seed, n_jobs=min(4, os.cpu_count() or 1)), \
               [early_stopping(40, verbose=False), log_evaluation(0)]
    if algo == 'catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=400, depth=4, learning_rate=.06,
                                  loss_function='Logloss', eval_metric='PRAUC',
                                  class_weights=[1.0, float(spw)], random_state=seed,
                                  od_type='Iter', od_wait=40, verbose=False, task_type=('GPU' if device == 'cuda' else 'CPU'))
    raise ValueError(algo)


def run_fold(algo, feats, train_rows, fold_valid, features, seed, device='cpu'):
    numeric_cols = FEATURE_SETS[feats][0]
    y_fold = train_rows.fraud_bool.astype('int8')
    pos = int(y_fold.sum())
    spw = (len(y_fold) - pos) / pos
    if algo == 'catboost_native':
        # Native categorical handling: no OHE; numeric medians; raw category strings.
        train = train_rows[features].copy()
        valid = fold_valid[features].copy()
        for col in numeric_cols:
            med = train[col].median()
            train[col] = train[col].fillna(med)
            valid[col] = valid[col].fillna(med)
        for col in CATEGORICAL:
            train[col] = train[col].astype('string').fillna('__missing__').astype(str)
            valid[col] = valid[col].astype('string').fillna('__missing__').astype(str)
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=400, depth=4, learning_rate=.06,
                                   loss_function='Logloss', eval_metric='PRAUC',
                                   class_weights=[1.0, float(spw)], random_state=seed,
                                   od_type='Iter', od_wait=40, verbose=False,
                                   cat_features=list(range(len(numeric_cols), len(numeric_cols) + len(CATEGORICAL))),
                                   task_type=('GPU' if device == 'cuda' else 'CPU'))
        model.fit(train, y_fold, eval_set=(valid, fold_valid.fraud_bool.astype('int8')), verbose=False)
        scores = model.predict_proba(valid)[:, 1]
    elif algo == 'lightgbm_fix':
        pre = make_preprocess(numeric_cols)
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        model = LGBMClassifier(n_estimators=400, max_depth=-1, num_leaves=31, learning_rate=.06,
                               subsample=.85, colsample_bytree=.85, reg_lambda=4,
                               is_unbalance=True, random_state=seed, n_jobs=min(4, os.cpu_count() or 1))
        x_tr = pre.fit_transform(train_rows[features])
        x_va = pre.transform(fold_valid[features])
        model.fit(x_tr, y_fold, eval_set=[(x_va, fold_valid.fraud_bool.astype('int8'))],
                  eval_metric='average_precision',
                  callbacks=[early_stopping(40, verbose=False), log_evaluation(0)])
        scores = model.predict_proba(x_va)[:, 1]
    else:
        pre = make_preprocess(numeric_cols)
        built = make_model(algo, spw, seed, device)
        model, callbacks = (built, None) if algo != 'lightgbm' else built
        x_tr = pre.fit_transform(train_rows[features])
        x_va = pre.transform(fold_valid[features])
        if algo == 'lightgbm':
            model.fit(x_tr, y_fold, eval_set=[(x_va, fold_valid.fraud_bool.astype('int8'))],
                      eval_metric='aucpr', callbacks=callbacks)
        elif algo == 'catboost':
            model.fit(x_tr, y_fold, eval_set=(x_va, fold_valid.fraud_bool.astype('int8')), verbose=False)
        else:
            model.fit(x_tr, y_fold, eval_set=[(x_va, fold_valid.fraud_bool.astype('int8'))], verbose=False)
        scores = model.predict_proba(x_va)[:, 1]
    threshold, _ = threshold_analysis(fold_valid.fraud_bool, scores, 100)
    m = metrics(fold_valid.fraud_bool, scores, threshold)
    m.update(ranking_metrics(fold_valid.fraud_bool, scores))
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algos', nargs='+', default=['xgboost', 'lightgbm', 'catboost'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    data = read_base(args.data)
    data = add_engineered(data)
    parts, boundaries = split_months(data)
    train_rows = parts[0]
    months = sorted(train_rows.month.unique().tolist())
    results = {}
    for feats in ['v1', 'v2']:
        features = FEATURE_SETS[feats][1]
        for algo in args.algos:
            key = f'{algo}_{feats}'
            results[key] = []
            for k in range(3, len(months)):
                fold_train = train_rows[train_rows.month.isin(months[:k])]
                fold_valid = train_rows[train_rows.month == months[k]]
                m = run_fold(algo, feats, fold_train, fold_valid, features, args.seed, args.device)
                results[key].append({'valid_month': months[k], 'metrics': m})
                print(f'{key} fold valid {months[k]}: PR-AUC {m["average_precision_pr_auc"]:.6f} '
                      f'F1 {m["f1"]:.6f}', flush=True)
    summary = {}
    for key, folds in results.items():
        mean = {k: round(float(np.mean([f['metrics'][k] for f in folds])), 6)
                for k in folds[0]['metrics'] if isinstance(folds[0]['metrics'][k], (int, float))}
        summary[key] = {'folds': [{k: (v if isinstance(v, (int, float)) else str(v)) for k, v in f['metrics'].items()}
                                   | {'valid_month': f['valid_month']} for f in folds],
                        'mean_metrics': mean}
    report = {'mode': 'research-algorithms', 'trained_at_utc': datetime.now(timezone.utc).isoformat(),
              'results': summary,
              'note': 'Nominal GPU task_type used where available; penultimate/final months never touched.'}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for key, r in summary.items():
        print(key, 'mean PR-AUC', r['mean_metrics']['average_precision_pr_auc'],
              'F1', r['mean_metrics']['f1'], flush=True)


if __name__ == '__main__':
    main()
