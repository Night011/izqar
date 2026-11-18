"""Research-only algorithm comparison for BAF under the calibrated protocol.

Every fold has three disjoint periods: train < calibration < external. The
number of boosting rounds and the operating threshold are chosen on the
calibration month; the external month is scored once with both frozen.

Why rounds are selected explicitly instead of with early stopping: the earlier
LightGBM arm used LightGBM early stopping on a class-weighted model. The
monitored metric was weighted logloss, which necessarily rises once the model
starts reweighting the classes, so the callback kept round 1 and LightGBM
discarded every later tree (``booster.current_iteration() == 1``). The reported
PR-AUC 0.052 was a one-tree model, not a LightGBM limitation. PR-AUC on the
calibration month is now the single selection rule for every algorithm.

No artifact is written; the penultimate and final months are never touched.

Each fold also records a second operating point: the calibration threshold is
translated into the share of the batch it flags, and that alert rate is
replayed on the external month (``external_metrics_capacity``). Replaying a
share uses only the score distribution, so it stays label-free. With
``--refit-on-calibration`` the model is additionally refitted on
train+calibration with the round count frozen, which models a pre-deployment
retrain without letting the calibration month drive any choice.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baf_protocol import (budget_alert_rate, calibrated_folds, evaluate_external,  # noqa: E402
                          evaluate_external_at_capacity, mean_external_metrics,
                          rows_fingerprint, select_operating_point)
from train_baf import (CATEGORICAL, FEATURE_SETS, VELOCITY, add_engineered, read_base,  # noqa: E402
                       split_months)
from train_ieee import file_provenance  # noqa: E402

ROUND_GRID_STEP = 25

LIGHTGBM_PARAMS = {
    # Unweighted model; rounds chosen on calibration PR-AUC.
    'lightgbm': dict(n_estimators=400, num_leaves=15, learning_rate=.06, subsample=.85,
                     colsample_bytree=.85, reg_lambda=4.0, min_child_samples=100),
    # Heavily regularised, class-weighted variant that won on calibration PR-AUC.
    'lightgbm_reg': dict(n_estimators=800, num_leaves=7, learning_rate=.03, subsample=.8,
                         colsample_bytree=.6, reg_lambda=20.0, min_child_samples=500),
}


def make_preprocess(numeric_cols):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True))])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20))])
    return ColumnTransformer([('numeric', numeric, numeric_cols),
                              ('categorical', categorical, CATEGORICAL)])


def _ohe_member(name, preprocess, feature_cols, params, max_rounds, predict):
    return {'name': name, 'max_rounds': max_rounds, 'params': params,
            'transform': lambda frame: preprocess.transform(frame[feature_cols]),
            'predict': predict}


def _xgboost_member(fit_rows, feature_cols, numeric_cols, *, seed, max_rounds, device, extra=None):
    from xgboost import XGBClassifier
    preprocess = make_preprocess(numeric_cols)
    y = fit_rows.fraud_bool.astype('int8')
    positive = int(y.sum())
    spw = (len(y) - positive) / positive
    params = dict(n_estimators=max_rounds, max_depth=4, learning_rate=.06, subsample=.85,
                  colsample_bytree=.85, reg_lambda=4.0, scale_pos_weight=round(spw, 6))
    params.update(extra or {})
    model = XGBClassifier(**params, objective='binary:logistic', tree_method='hist',
                          eval_metric='aucpr', random_state=seed, device=device,
                          n_jobs=min(4, os.cpu_count() or 1))
    model.fit(preprocess.fit_transform(fit_rows[feature_cols]), y, verbose=False)
    return _ohe_member('xgboost', preprocess, feature_cols, params, max_rounds,
                       lambda matrix, rounds: model.predict_proba(matrix, iteration_range=(0, rounds))[:, 1])


def _catboost_member(fit_rows, feature_cols, numeric_cols, *, seed, max_rounds, device,
                     native=False, extra=None):
    from catboost import CatBoostClassifier
    y = fit_rows.fraud_bool.astype('int8')
    positive = int(y.sum())
    spw = (len(y) - positive) / positive
    model_params = dict(iterations=max_rounds, depth=4, learning_rate=.06,
                        class_weights=[1.0, round(float(spw), 6)])
    model_params.update({k: v for k, v in (extra or {}).items() if k != 'iterations'})
    params = dict(model_params, native_categorical=native)
    if native:
        medians = {col: float(fit_rows[col].median()) for col in numeric_cols}

        def transform(frame):
            out = frame[feature_cols].copy()
            for col in numeric_cols:
                out[col] = out[col].fillna(medians[col])
            for col in CATEGORICAL:
                out[col] = out[col].astype('string').fillna('__missing__').astype(str)
            return out

        cat_features = list(range(len(numeric_cols), len(numeric_cols) + len(CATEGORICAL)))
        model = CatBoostClassifier(**model_params, loss_function='Logloss', random_state=seed,
                                   verbose=False, allow_writing_files=False, cat_features=cat_features,
                                   task_type=('GPU' if device == 'cuda' else 'CPU'))
        model.fit(transform(fit_rows), y, verbose=False)
    else:
        preprocess = make_preprocess(numeric_cols)
        model = CatBoostClassifier(**model_params, loss_function='Logloss', random_state=seed,
                                   verbose=False, allow_writing_files=False,
                                   task_type=('GPU' if device == 'cuda' else 'CPU'))
        model.fit(preprocess.fit_transform(fit_rows[feature_cols]), y, verbose=False)

        def transform(frame):
            return preprocess.transform(frame[feature_cols])

    return {'name': 'catboost_native' if native else 'catboost', 'max_rounds': max_rounds,
            'params': params, 'transform': transform,
            'predict': lambda matrix, rounds: model.predict_proba(
                matrix, ntree_start=0, ntree_end=rounds)[:, 1]}


def _lightgbm_member(algo, fit_rows, feature_cols, numeric_cols, *, seed, device, extra=None):
    from lightgbm import LGBMClassifier
    preprocess = make_preprocess(numeric_cols)
    y = fit_rows.fraud_bool.astype('int8')
    params = dict(LIGHTGBM_PARAMS[algo])
    if algo == 'lightgbm_reg':
        params['scale_pos_weight'] = round((len(y) - int(y.sum())) / int(y.sum()), 6)
    params.update(extra or {})
    max_rounds = int(params['n_estimators'])
    model = LGBMClassifier(**params, random_state=seed, n_jobs=min(4, os.cpu_count() or 1), verbose=-1)
    model.fit(preprocess.fit_transform(fit_rows[feature_cols]), y)
    params['device'] = 'cpu'
    return _ohe_member(algo, preprocess, feature_cols, params, max_rounds,
                       lambda matrix, rounds: model.predict_proba(matrix, num_iteration=rounds)[:, 1])


def _lightgbm_native_member(algo, fit_rows, feature_cols, numeric_cols, *, seed, device, extra=None):
    """LightGBM with native categoricals and raw NaN, i.e. the published recipe.

    Category codes are frozen on the fit rows; unseen levels map to NaN, which
    LightGBM treats as missing. Numeric NaN is passed through untouched instead
    of being imputed, so missingness stays a split direction.
    """
    from lightgbm import LGBMClassifier
    y = fit_rows.fraud_bool.astype('int8')
    params = dict(LIGHTGBM_PARAMS['lightgbm'])
    if algo == 'lightgbm_native_reg':
        params = dict(LIGHTGBM_PARAMS['lightgbm_reg'])
        params['scale_pos_weight'] = round((len(y) - int(y.sum())) / int(y.sum()), 6)
    params.update(extra or {})
    max_rounds = int(params['n_estimators'])
    codes = {name: {value: index for index, value in enumerate(
        sorted(fit_rows[name].astype('string').fillna('__missing__').unique()))}
        for name in CATEGORICAL}

    def transform(frame):
        out = frame[feature_cols].copy()
        for name in CATEGORICAL:
            out[name] = out[name].astype('string').fillna('__missing__').map(codes[name])
        return out.astype('float32')

    model = LGBMClassifier(**params, random_state=seed, n_jobs=min(4, os.cpu_count() or 1),
                           verbose=-1)
    model.fit(transform(fit_rows), y, categorical_feature=CATEGORICAL)
    return {'name': algo, 'max_rounds': max_rounds, 'params': params, 'transform': transform,
            'predict': lambda matrix, rounds: model.predict_proba(matrix, num_iteration=rounds)[:, 1]}


def build_members(algo, fit_rows, feature_cols, numeric_cols, *, seed, device, max_rounds,
                  extra_params=None):
    """Return (members, combination). A member is a fitted model plus a frozen-round scorer."""
    if algo == 'xgboost':
        return [_xgboost_member(fit_rows, feature_cols, numeric_cols, seed=seed,
                                max_rounds=max_rounds, device=device,
                                extra=extra_params)], None
    if algo in {'catboost', 'catboost_native'}:
        return [_catboost_member(fit_rows, feature_cols, numeric_cols, seed=seed,
                                 max_rounds=max_rounds, device=device,
                                 native=algo == 'catboost_native', extra=extra_params)], None
    if algo in LIGHTGBM_PARAMS:
        return [_lightgbm_member(algo, fit_rows, feature_cols, numeric_cols, seed=seed,
                                 device=device, extra=extra_params)], None
    if algo in {'lightgbm_native', 'lightgbm_native_reg'}:
        return [_lightgbm_native_member(algo, fit_rows, feature_cols, numeric_cols, seed=seed,
                                        device=device, extra=extra_params)], None
    if algo == 'ensemble':
        return [_xgboost_member(fit_rows, feature_cols, numeric_cols, seed=seed,
                                max_rounds=max_rounds, device=device),
                _catboost_member(fit_rows, feature_cols, numeric_cols, seed=seed,
                                 max_rounds=max_rounds, device=device, native=True)], 'rank_average'
    raise ValueError(f'unknown algorithm {algo}')


def _rank_average(score_list):
    ranked = []
    for scores in score_list:
        order = np.argsort(np.argsort(scores, kind='stable'), kind='stable')
        ranked.append(order / max(len(order) - 1, 1))
    return np.mean(ranked, axis=0)


def _combine(score_list, combination):
    if combination is None:
        return score_list[0]
    if combination == 'rank_average':
        return _rank_average(score_list)
    raise ValueError(f'unknown combination {combination}')


def _select_rounds(y_calibration, matrix, predict, max_rounds: int):
    """Pick the round count maximising calibration PR-AUC; no external labels involved."""
    grid = list(range(ROUND_GRID_STEP, max_rounds + 1, ROUND_GRID_STEP))
    sweep, best_rounds, best_score = [], grid[0], -1.0
    for rounds in grid:
        score = float(average_precision_score(y_calibration, predict(matrix, rounds)))
        sweep.append({'rounds': rounds, 'calibration_pr_auc': round(score, 6)})
        if score > best_score:
            best_rounds, best_score = rounds, score
    return best_rounds, round(best_score, 6), sweep


def evaluate_fold(algo, fit_rows, calibration_rows, external_rows, feature_cols,
                  numeric_cols, *, seed, device, max_rounds, budget, refit=False,
                  extra_params=None, return_scores=False) -> dict:
    started = time.perf_counter()
    members, combination = build_members(algo, fit_rows, feature_cols, numeric_cols,
                                         seed=seed, device=device, max_rounds=max_rounds,
                                         extra_params=extra_params)
    fit_seconds = time.perf_counter() - started
    y_calibration = calibration_rows.fraud_bool.to_numpy()
    y_external = external_rows.fraud_bool.to_numpy()

    matrices = {'calibration': [], 'external': []}
    rounds, calibration = {}, {}
    for member in members:
        x_calibration = member['transform'](calibration_rows)
        x_external = member['transform'](external_rows)
        chosen, best_score, sweep = _select_rounds(y_calibration, x_calibration,
                                                  member['predict'], member['max_rounds'])
        rounds[member['name']] = chosen
        calibration[member['name']] = {'selected_rounds': chosen,
                                       'calibration_pr_auc': best_score,
                                       'sweep': sweep}
        matrices['calibration'].append(member['predict'](x_calibration, chosen))
        matrices['external'].append(member['predict'](x_external, chosen))

    calibration_scores = _combine(matrices['calibration'], combination)
    external_scores = _combine(matrices['external'], combination)
    threshold, selection = select_operating_point(y_calibration, calibration_scores, budget)
    alert_rate = budget_alert_rate(y_calibration, calibration_scores, budget)
    report = {
        'rounds': rounds, 'fit_seconds': round(float(fit_seconds), 3),
        'selected_threshold': round(float(threshold), 6),
        'selection': selection, 'calibration': calibration,
        'params': {member['name']: member['params'] for member in members},
        'combination': combination,
        'capacity_alert_rate': round(float(alert_rate), 6),
        'external_metrics': evaluate_external(y_external, external_scores, threshold),
        'external_metrics_capacity': evaluate_external_at_capacity(
            y_external, external_scores, alert_rate),
    }
    if return_scores:
        report['scores'] = {'calibration': np.asarray(calibration_scores, dtype='float32'),
                            'external': np.asarray(external_scores, dtype='float32')}

    if refit:
        # Deploy-before-external retrain: rounds stay frozen, so the calibration
        # month only contributes rows (it never re-selects anything).
        if combination is not None:
            raise ValueError('refit is defined for single-model arms only')
        refit_started = time.perf_counter()
        chosen = rounds[members[0]['name']]
        refit_members, _ = build_members(algo, pd.concat([fit_rows, calibration_rows]),
                                         feature_cols, numeric_cols, seed=seed,
                                         device=device, max_rounds=chosen,
                                         extra_params=extra_params)
        member = refit_members[0]
        refit_scores = member['predict'](member['transform'](external_rows), chosen)
        report['refit'] = {'rounds': chosen, 'fit_seconds': round(float(time.perf_counter() - refit_started), 3),
                           'train_rows': int(len(fit_rows) + len(calibration_rows)),
                           'note': 'model refitted on train+calibration with frozen rounds; '
                                   'operating point replayed as the calibration alert rate'}
        report['external_metrics_refit_capacity'] = evaluate_external_at_capacity(
            y_external, refit_scores, alert_rate)
        report['external_metrics_refit_capacity']['threshold_source'] = (
            'refit model, frozen alert rate replayed on external scores (no labels)')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algos', nargs='+',
                        default=['xgboost', 'catboost', 'catboost_native', 'lightgbm', 'lightgbm_reg', 'ensemble'])
    parser.add_argument('--feature-sets', nargs='+', default=['v1'], choices=['v1', 'v2', 'v1v', 'v1x'])
    parser.add_argument('--max-rounds', type=int, default=400)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--refit-on-calibration', action='store_true',
                        help='also score a model refitted on train+calibration with frozen rounds')
    parser.add_argument('--params-json', type=Path,
                        help='tuned hyperparameters per algorithm, as written by research_baf_tune.py')
    parser.add_argument('--mask-balcon-missing', action='store_true',
                        help='treat negative intended_balcon_amount as missing, per the BAF datasheet')
    parser.add_argument('--scores-dir', type=Path,
                        help='save out-of-fold external scores as .npy for later ensembling')
    parser.add_argument('--scores-tag', default='',
                        help='suffix for saved score files, e.g. a seed repeat')
    parser.add_argument('--max-false-alerts-per-10000', type=float, default=100.0)
    args = parser.parse_args()

    tuned = json.loads(args.params_json.read_text(encoding='utf-8')) if args.params_json else {}
    extra = VELOCITY if ('v1v' in args.feature_sets or 'v1x' in args.feature_sets) else ()
    data = add_engineered(read_base(args.data, extra))
    if args.mask_balcon_missing:
        data['intended_balcon_amount'] = data['intended_balcon_amount'].mask(
            data['intended_balcon_amount'] < 0)
    parts, boundaries = split_months(data)
    train_rows = parts[0]
    months = sorted(train_rows.month.unique().tolist())
    folds = calibrated_folds(months)
    results = {}
    for feats in args.feature_sets:
        feature_cols, numeric_cols = FEATURE_SETS[feats][1], FEATURE_SETS[feats][0]
        for algo in args.algos:
            key = f'{algo}_{feats}{args.scores_tag}'
            fold_reports = []
            for fold in folds:
                fit_rows = train_rows[train_rows.month.isin(fold['train_months'])]
                calibration_rows = train_rows[train_rows.month == fold['calibration_month']]
                external_rows = train_rows[train_rows.month == fold['external_month']]
                evaluation = evaluate_fold(
                    algo, fit_rows, calibration_rows, external_rows, feature_cols,
                    numeric_cols, seed=args.seed, device=args.device, max_rounds=args.max_rounds,
                    budget=args.max_false_alerts_per_10000, refit=args.refit_on_calibration,
                    extra_params=(tuned.get('best_params') if 'best_params' in tuned
                                  else tuned.get(algo)),
                    return_scores=bool(args.scores_dir))
                if args.scores_dir:
                    args.scores_dir.mkdir(parents=True, exist_ok=True)
                    stem = f"{key}_m{fold['external_month']}"
                    for split in ('calibration', 'external'):
                        np.save(args.scores_dir / f'{stem}_{split}.npy', evaluation['scores'][split])
                    np.save(args.scores_dir / f'{stem}_labels.npy',
                            external_rows.fraud_bool.to_numpy(dtype='int8'))
                    np.save(args.scores_dir / f'{stem}_cal_labels.npy',
                            calibration_rows.fraud_bool.to_numpy(dtype='int8'))
                    del evaluation['scores']
                fold_reports.append({
                    'train_months': fold['train_months'],
                    'calibration_month': fold['calibration_month'],
                    'external_month': fold['external_month'],
                    'rows': {'train': len(fit_rows), 'calibration': len(calibration_rows),
                             'external': len(external_rows)},
                    'checksums': {'train': rows_fingerprint(fit_rows),
                                  'calibration': rows_fingerprint(calibration_rows),
                                  'external': rows_fingerprint(external_rows)},
                    **evaluation,
                })
                external = evaluation['external_metrics']
                print(f"{key} fold external {fold['external_month']}: "
                      f"PR-AUC {external['average_precision_pr_auc']:.6f} "
                      f"F1 {external['f1']:.6f} "
                      f"FP/10k {external['false_alerts_per_10000_operations']:.1f} "
                      f"rounds {evaluation['rounds']} ({evaluation['fit_seconds']:.1f}s)", flush=True)
                capacity = evaluation['external_metrics_capacity']
                line = (f"{key} fold external {fold['external_month']} capacity: "
                        f"alert rate {evaluation['capacity_alert_rate']:.4f} "
                        f"recall {capacity['recall']:.4f} "
                        f"FP/10k {capacity['false_alerts_per_10000_operations']:.1f}")
                refit = evaluation.get('external_metrics_refit_capacity')
                if refit:
                    line += (f" | refit: PR-AUC {refit['average_precision_pr_auc']:.6f} "
                             f"recall {refit['recall']:.4f} "
                             f"FP/10k {refit['false_alerts_per_10000_operations']:.1f}")
                print(line, flush=True)
            results[key] = {'folds': fold_reports, 'mean_metrics': mean_external_metrics(fold_reports),
                            'mean_metrics_capacity': mean_external_metrics(
                                fold_reports, 'external_metrics_capacity'),
                            'mean_metrics_refit_capacity': mean_external_metrics(
                                fold_reports, 'external_metrics_refit_capacity')}
            print(key, 'mean PR-AUC', results[key]['mean_metrics']['average_precision_pr_auc'], flush=True)

    report = {'mode': 'research-algorithms-calibrated', 'protocol': 'calibrated',
              'trained_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': args.seed,
              'max_false_alerts_per_10000': args.max_false_alerts_per_10000,
              'data_provenance': file_provenance(args.data),
              'split_method': ('expanding month folds: train < calibration < external, strictly disjoint; '
                               'rounds and threshold fixed on calibration; penultimate/final months untouched'),
              'results': results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote', args.output)


if __name__ == '__main__':
    main()
