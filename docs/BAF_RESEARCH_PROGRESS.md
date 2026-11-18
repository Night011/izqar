# BAF Fraud Detection — Research Progress

Worktree: `C:\Users\a\Documents\IZQAR-BAF`, branch `feat/baf-research` (from `origin/main` = b6c8c5b).

## Audit (2026-11-18)

- Trainer `backend/scripts/train_baf.py`: whitelist 15 numerics + 5 categoricals; months split — train [<penultimate-1], validation = penultimate month, test = last month (final; evaluated only in `final` mode; `experiment` mode omits test metrics). Sentinel -1 → NaN for 4 features. XGBoost(max_depth=4, lr=.06, n=180, spw=(neg/pos)*multiplier) + LogisticRegression baseline with liblinear.
- Data: `sgpjesus/bank-account-fraud-dataset-neurips-2022` (Kaggle, 558MB; Base.csv 213MB) — not stored in repo. Download in progress to `backend/data/baf/` (gitignored).
- Confirmed metrics on disk `baf_base_v1_portable.metrics.json`:
  - validation: PR-AUC 0.163963, ROC 0.8854, F1 0.2301, FP/10k 99.9 (threshold 0.9142)
  - test (previously used): PR-AUC 0.199692, ROC 0.8913, F1 0.241141, FP/10k 75.5 — matches historical orienteers. Test is a historical benchmark, NOT an independent confirmation.
- Prior candidates (validation-only, threshold budget 100 FP/10k): weight075 0.164185, recall 0.162608, weight075+early_stop 0.165998 — all within noise of baseline 0.163963. No wins yet; class-weight tweaks insufficient.
- Leakage exclusions documented in artifact: fraud_bool, month, device_fraud_count, days_since_request, credit_risk_score, unverified aggregates.
- Plan: engineered application-time features + month-expanding research folds (temporal, like IEEE) + XGBoost/LightGBM/CatBoost; internal folds drive selection; previously used test not touched for tuning.

## IEEE-CIS tech debt (carried forward)

- Fold 3 FP/10k = 106.9 vs budget 100; test FP/10k = 94.0 (129.6 for ieee_final_v2).
- IEEE results on the previously used test are not independent confirmation.
- Production inference of the IEEE candidate requires extra verification before any deploy.
- GitHub Actions blocked by account billing lock (all branches, 2–4s, zero steps); 144 local tests pass.
- Revert note: to roll back the IBM+IEEE merge use targeted `git revert` on the specific merge commit — never reset main to 0d2e9ea (would lose both agents' work).

## Log
