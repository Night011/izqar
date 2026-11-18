"""Limited, reproducible hyperparameter search for the BAF calibrated folds.

The objective is PR-AUC on month 2 with the model fitted on months 0-1. Month 2
is only ever a calibration month, never an external fold, so the search cannot
see the labels of any fold it is later reported on. The winning configuration is
frozen and re-evaluated on folds m3/m4/m5 by research_baf_algorithms.py.

Spaces are deliberately narrow (single-digit hyperparameters, ~30 trials) because
the dataset gives no room for wide searches: 3 folds with ~1.4k frauds each.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_baf_algorithms import build_members  # noqa: E402
from train_baf import FEATURE_SETS, VELOCITY, add_engineered, read_base, split_months  # noqa: E402

ROUNDS = 400


def suggest(trial, algo: str) -> dict:
    if algo == 'lightgbm':
        return {
            'num_leaves': trial.suggest_int('num_leaves', 12, 96, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.12, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 500, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'subsample_freq': 1,
            'n_estimators': trial.suggest_int('n_estimators', 150, 600),
        }
    if algo == 'catboost_native':
        return {
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.12, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 30.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.0, 3.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        }
    raise ValueError(f'no search space for {algo}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--algo', default='lightgbm', choices=['lightgbm', 'catboost_native'])
    parser.add_argument('--feature-set', default='v1v', choices=['v1v', 'v1x', 'v1'])
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    extra = VELOCITY if args.feature_set in {'v1v', 'v1x'} else ()
    data = add_engineered(read_base(args.data, extra))
    train_rows = split_months(data)[0][0]
    fit_rows = train_rows[train_rows.month < 2]
    calibration_rows = train_rows[train_rows.month == 2]
    y_calibration = calibration_rows.fraud_bool.to_numpy()
    numeric_cols, feature_cols = FEATURE_SETS[args.feature_set]

    def objective(trial):
        params = suggest(trial, args.algo)
        members, _ = build_members(args.algo, fit_rows, feature_cols, numeric_cols,
                                   seed=args.seed, device='cpu', max_rounds=ROUNDS,
                                   extra_params=params)
        member = members[0]
        scores = member['predict'](member['transform'](calibration_rows), ROUNDS)
        score = float(average_precision_score(y_calibration, scores))
        print(f"trial {trial.number}: {score:.6f} {params}", flush=True)
        return score

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=8))
    study.optimize(objective, n_trials=args.trials)

    report = {
        'mode': 'baf-hyperparameter-search',
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'algo': args.algo, 'feature_set': args.feature_set, 'seed': args.seed,
        'search_data': 'fit months [0, 1] -> PR-AUC on calibration month 2 (never an external fold)',
        'rounds': ROUNDS, 'trials': args.trials,
        'best_value': round(float(study.best_value), 6),
        'best_params': study.best_params,
        'history': [{'number': t.number, 'value': None if t.value is None else round(float(t.value), 6),
                     'params': t.params} for t in study.trials],
        'baseline_value': round(float(average_precision_score(
            y_calibration, _default_scores(args, fit_rows, calibration_rows, numeric_cols,
                                           feature_cols))), 6)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"best {report['best_value']:.6f} vs default {report['baseline_value']:.6f}")
    print('wrote', args.output)


def _default_scores(args, fit_rows, calibration_rows, numeric_cols, feature_cols):
    members, _ = build_members(args.algo, fit_rows, feature_cols, numeric_cols,
                               seed=args.seed, device='cpu', max_rounds=ROUNDS)
    member = members[0]
    return member['predict'](member['transform'](calibration_rows), ROUNDS)


if __name__ == '__main__':
    main()
