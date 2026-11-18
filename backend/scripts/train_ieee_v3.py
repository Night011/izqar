"""Train an improved IEEE-CIS transaction classifier (experimental candidate).

Full C/D/V feature set + engineered amount/time/email/missingness features and
*causal* frequency features: each row only sees occurrences of its keys in
strictly earlier rows (chronological cumcount). No future information is used,
not even inside the training period.

Evaluation modes:
  * research (default): 3 expanding chronological folds inside the first 85% of
    the data. The final 15% tail is never evaluated or needed; no production
    artifact is written. Use this for hyperparameter/feature comparisons.
  * final: single chronological 70/15/15 split; threshold from validation only;
    the previously used test set is evaluated once and is a *historical
    benchmark*, not an independent sealed holdout.

Protocol: chronological order by TransactionDT; threshold tuned on validation
only; no synthetic simulator data; artifact format v1 (compatible with
app.ml.predict_ieee via IZQAR_MODEL_PATH).
"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

BASE_NUMERIC = [
    'TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2',
    'dist1', 'dist2',
] + [f'C{i}' for i in range(1, 15)] + [f'D{i}' for i in range(1, 16)] + [f'V{i}' for i in range(1, 340)]
CATEGORICAL = [
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'DeviceType', 'DeviceInfo', 'id_31',
]
ENGINEERED_NUMERIC = ['amt_log', 'amt_cents', 'dt_hour', 'dt_day', 'email_match',
                      'v_missing', 'd_missing',
                      'card1_freq', 'card1_card5_freq', 'card2_freq', 'addr1_freq',
                      'p_email_freq', 'device_freq']
FREQ_SOURCES = {'card1_freq': ['card1'], 'card1_card5_freq': ['card1', 'card5'],
                'card2_freq': ['card2'], 'addr1_freq': ['addr1'],
                'p_email_freq': ['P_emaildomain'], 'device_freq': ['DeviceInfo']}
# Expanding folds inside the first 85% of the chronological data (the tail is
# physically excluded before any processing). Fractions are relative to the
# truncated pool and match the original .40/.55/.70/.85 boundaries of full data.
RESEARCH_CEILING = .85
RESEARCH_FOLDS = ((.47, .647), (.647, .824), (.824, 1.0))


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    amt = pd.to_numeric(df['TransactionAmt'], errors='coerce')
    df['amt_log'] = np.log1p(amt)
    df['amt_cents'] = (amt * 100) % 100
    dt = pd.to_numeric(df['TransactionDT'], errors='coerce')
    df['dt_hour'] = (dt // 3600) % 24
    df['dt_day'] = dt // 86400
    p, r = df.get('P_emaildomain'), df.get('R_emaildomain')
    df['email_match'] = ((p.astype('string') == r.astype('string')) & p.notna() & r.notna()).astype('float32')
    vcols = [c for c in df.columns if c.startswith('V') and c[1:].isdigit()]
    dcols = [f'D{i}' for i in range(1, 16)]
    df['v_missing'] = df[vcols].isna().sum(axis=1).astype('float32')
    df['d_missing'] = df[dcols].isna().sum(axis=1).astype('float32')
    return df


def _train_freq_state(train_rows: pd.DataFrame) -> dict[str, dict[str, int]]:
    state: dict[str, dict[str, int]] = {}
    for name, cols in FREQ_SOURCES.items():
        if any(col not in train_rows.columns for col in cols):
            continue
        key = freq_key(train_rows, cols)
        state[name] = {str(k): int(v) for k, v in key.value_counts().items()}
    return state


def freq_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if len(cols) > 1:
        return df[cols].astype('string').fillna('__missing__').agg('_'.join, axis=1)
    return df[cols[0]].astype('string').fillna('__missing__')


def add_causal_freq(df: pd.DataFrame, scale: str = 'linear') -> pd.DataFrame:
    """Prior-occurrence counts; strictly causal because df is TransactionDT-sorted.

    scale='linear' keeps raw prior counts; 'log' applies log1p to compress heavy tails.
    """
    df = df.copy()
    for name, cols in FREQ_SOURCES.items():
        if any(col not in df.columns for col in cols):
            continue
        key = freq_key(df, cols)
        counts = key.groupby(key).cumcount().astype('float32')
        df[name] = np.log1p(counts) if scale == 'log' else counts
    return df


def select_feature_lists(columns) -> tuple[list[str], list[str], list[str]]:
    available = set(columns)
    nums = [c for c in BASE_NUMERIC + ENGINEERED_NUMERIC if c in available]
    cats = [c for c in CATEGORICAL if c in available]
    return nums + cats, nums, cats


def prepare_columns(train_rows, valid_rows, test_rows, nums, cats):
    for col in nums:
        for part in (train_rows, valid_rows, test_rows):
            part[col] = pd.to_numeric(part[col], errors='coerce').astype('float32')
    for col in cats:
        for part in (train_rows, valid_rows, test_rows):
            part[col] = part[col].astype('string').fillna('__missing__').astype(str)


def build_pipeline(nums, cats, n_estimators, seed, negatives, positives, device='cpu'):
    transform = ColumnTransformer([
        ('numeric', 'passthrough', nums),
        ('category', OneHotEncoder(handle_unknown='ignore', min_frequency=100,
                                   max_categories=60), cats),
    ], sparse_threshold=.3)
    classifier = XGBClassifier(
        n_estimators=n_estimators, max_depth=5, learning_rate=.05, subsample=.8,
        colsample_bytree=.6, reg_lambda=4, objective='binary:logistic',
        tree_method='hist', eval_metric='aucpr', random_state=seed,
        n_jobs=min(8, os.cpu_count() or 1), scale_pos_weight=float(negatives / max(positives, 1)),
        early_stopping_rounds=60, device=device,
    )
    return transform, classifier


def fit_on(train_rows, valid_rows, n_estimators, seed, device='cpu'):
    features, nums, cats = select_feature_lists(train_rows.columns)
    prepare_columns(train_rows, valid_rows, train_rows.iloc[:0], nums, cats)
    positives = int(train_rows['isFraud'].sum())
    negatives = len(train_rows) - positives
    transform, classifier = build_pipeline(nums, cats, n_estimators, seed, negatives, positives, device)
    x_train, y_train = train_rows[features], train_rows['isFraud'].astype(int)
    x_valid, y_valid = valid_rows[features], valid_rows['isFraud'].astype(int)
    x_train_t = transform.fit_transform(x_train)
    x_valid_t = transform.transform(x_valid)
    classifier.fit(x_train_t, y_train, eval_set=[(x_valid_t, y_valid)], verbose=False)
    pipeline = Pipeline([('preprocessing', transform), ('model', classifier)])
    return pipeline, features, nums, cats, getattr(classifier, 'best_iteration', None)


def metrics(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    order = np.argsort(-scores)
    y_sorted = np.asarray(y_true)[order]
    p_at = lambda k: float(y_sorted[:k].mean()) if len(y_sorted) >= k else float(y_sorted.mean())
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'average_precision_pr_auc': round(float(average_precision_score(y_true, scores)), 6),
        'roc_auc': round(float(roc_auc_score(y_true, scores)), 6),
        'precision': round(float(precision_score(y_true, pred, zero_division=0)), 6),
        'recall': round(float(recall_score(y_true, pred, zero_division=0)), 6),
        'f1': round(float(f1_score(y_true, pred, zero_division=0)), 6),
        'precision_at_100': round(p_at(100), 6),
        'precision_at_1000': round(p_at(1000), 6),
        'false_positive_rate': round(float(fp / max(fp + tn, 1)), 6),
        'false_negative_rate': round(float(fn / max(fn + tp, 1)), 6),
        'false_alerts_per_10000_operations': round(float(fp / max(len(y_true), 1) * 10000), 3),
        'false_negatives': int(fn),
        'confusion_matrix_tn_fp_fn_tp': [int(tn), int(fp), int(fn), int(tp)],
        'cases': int(len(y_true)),
        'fraud_cases': int(np.sum(y_true)),
    }


def choose_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def load_sorted(data_dir: Path, max_rows: int | None) -> pd.DataFrame:
    tx_path = data_dir / 'train_transaction.csv'
    id_path = data_dir / 'train_identity.csv'
    if not tx_path.exists():
        raise FileNotFoundError(f'Missing {tx_path}. Download IEEE-CIS from Kaggle.')
    data = pd.read_csv(tx_path, nrows=max_rows, low_memory=False)
    if id_path.exists():
        identity = pd.read_csv(id_path, low_memory=False)
        data = data.merge(identity, on='TransactionID', how='left', validate='one_to_one')
    data = data.sort_values(['TransactionDT', 'TransactionID'], kind='stable').reset_index(drop=True)
    for label, part in [('all', data)]:
        if part['isFraud'].nunique() != 2:
            raise ValueError(f'{label} must contain both classes.')
    return data


def build_features(data: pd.DataFrame, freq_mode: str) -> pd.DataFrame:
    data = add_engineered(data)
    if freq_mode == 'causal':
        data = add_causal_freq(data, scale='linear')
    elif freq_mode == 'causal_log':
        data = add_causal_freq(data, scale='log')
    return data


def run_research(data: pd.DataFrame, args, t0: float, excluded_tail_rows: int):
    n = len(data)
    fold_reports = []
    for train_frac, valid_frac in RESEARCH_FOLDS:
        train_end, valid_end = int(n * train_frac), int(n * valid_frac)
        train_rows = data.iloc[:train_end].copy()
        valid_rows = data.iloc[train_end:valid_end].copy()
        for label, part in [('train', train_rows), ('validation', valid_rows)]:
            if part['isFraud'].nunique() != 2:
                raise ValueError(f'fold {train_frac}-{valid_frac}: {label} split must contain both classes.')
        pipeline, features, nums, cats, best_iteration = fit_on(train_rows, valid_rows,
                                                                args.estimators, args.seed, args.device)
        report_features = features
        y_valid = valid_rows['isFraud'].astype(int).to_numpy()
        scores = pipeline.predict_proba(valid_rows[features])[:, 1]
        threshold = choose_threshold(y_valid, scores)
        fold_reports.append({'train_frac': train_frac, 'valid_frac': valid_frac,
                             'train_rows': len(train_rows), 'valid_rows': len(valid_rows),
                             'best_iteration': int(best_iteration) if best_iteration is not None else None,
                             'threshold': round(threshold, 6),
                             'metrics': metrics(y_valid, scores, threshold)})
        print(f"fold train<{train_frac} valid[{train_frac},{valid_frac}): "
              f"PR-AUC {fold_reports[-1]['metrics']['average_precision_pr_auc']:.6f} "
              f"F1 {fold_reports[-1]['metrics']['f1']:.6f}", flush=True)
    keys = fold_reports[0]['metrics'].keys()
    mean_metrics = {k: round(float(np.mean([r['metrics'][k] for r in fold_reports])), 6)
                    for k in keys if isinstance(fold_reports[0]['metrics'][k], (int, float))}
    report = {'mode': 'research', 'freq_mode': args.freq,
              'trained_at_utc': datetime.now(timezone.utc).isoformat(),
              'folds': fold_reports, 'mean_metrics': mean_metrics,
              'features_used': report_features,
              'excluded_tail_rows': excluded_tail_rows,
              'note': 'Final 15% tail was physically excluded before engineering; labels, features and statistics of the tail were never used.',
              'train_seconds': round(time.time() - t0, 1)}
    return report


def run_final(data: pd.DataFrame, args, t0: float):
    n = len(data)
    first, second = int(n * .70), int(n * .85)
    train_rows, valid_rows, test_rows = data.iloc[:first].copy(), data.iloc[first:second].copy(), data.iloc[second:].copy()
    for label, part in [('train', train_rows), ('validation', valid_rows), ('test', test_rows)]:
        if part['isFraud'].nunique() != 2:
            raise ValueError(f'{label} split must contain both classes.')
    pipeline, features, nums, cats, best_iteration = fit_on(train_rows, valid_rows,
                                                            args.estimators, args.seed, args.device)
    prepare_columns(train_rows.iloc[:0], valid_rows, test_rows, nums, cats)
    y_valid = valid_rows['isFraud'].astype(int).to_numpy()
    y_test = test_rows['isFraud'].astype(int).to_numpy()
    validation_scores = pipeline.predict_proba(valid_rows[features])[:, 1]
    threshold = choose_threshold(y_valid, validation_scores)
    test_scores = pipeline.predict_proba(test_rows[features])[:, 1]
    validation_metrics = metrics(y_valid, validation_scores, threshold)
    test_metrics = metrics(y_test, test_scores, threshold)
    artifact = {
        'format_version': 1,
        'model_version': 'ieee-cis-xgboost-v4-candidate',
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'IEEE-CIS Fraud Detection (Kaggle competition)',
        'pipeline': pipeline,
        'features': features,
        'numerical_features': nums,
        'categorical_features': cats,
        'freq_mode': args.freq,
        'freq_sources': FREQ_SOURCES if args.freq != 'none' else {},
        'freq_scale': 'log' if args.freq == 'causal_log' else 'linear',
        'freq_state': _train_freq_state(train_rows) if args.freq != 'none' else {},
        'threshold': threshold,
        'validation_metrics': validation_metrics,
        'test_metrics': test_metrics,
        'split_method': '70/15/15 chronological TransactionDT ascending; test tail is a historical benchmark, not a sealed holdout',
        'best_iteration': int(best_iteration) if best_iteration is not None else None,
        'train_seconds': round(time.time() - t0, 1),
        'warning': 'Research dataset; test used previously for benchmarking; scores not calibrated for deployment.',
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    report = {k: v for k, v in artifact.items() if k != 'pipeline'}
    report['threshold'] = round(threshold, 6)
    report['mode'] = 'final'
    output.with_suffix('.metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'model_path': str(output), 'best_iteration': artifact['best_iteration'],
                      'train_seconds': artifact['train_seconds'],
                      'validation': validation_metrics, 'test': test_metrics}, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description='Train improved IEEE-CIS candidate')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('models/experiments/ieee_v4_candidate.joblib'))
    parser.add_argument('--estimators', type=int, default=1500)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=['research', 'final'], default='research')
    parser.add_argument('--freq', choices=['causal', 'causal_log', 'none'], default='causal')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    t0 = time.time()
    data = load_sorted(args.data_dir, args.max_rows)
    excluded_tail_rows = 0
    if args.mode == 'research':
        cutoff = int(len(data) * RESEARCH_CEILING)
        excluded_tail_rows = len(data) - cutoff
        data = data.iloc[:cutoff].reset_index(drop=True)  # physically exclude the tail before any processing
    data = build_features(data, args.freq)
    if args.mode == 'research':
        report = run_research(data, args, t0, excluded_tail_rows)
        out = Path(str(args.output).replace('.joblib', '.cv.metrics.json')) if str(args.output).endswith('.joblib') \
            else Path(args.output).with_suffix('.cv.metrics.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report['mean_metrics'], ensure_ascii=False, indent=2))
        print(f'research report: {out}')
    else:
        run_final(data, args, t0)


if __name__ == '__main__':
    main()
