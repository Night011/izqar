# IEEE-CIS Research Progress

Branch: `feat/ieee-cis-research` (worktree `C:\Users\a\Documents\IZQAR-IEEE`).
Data read-only from `C:\Users\a\Documents\IZQAR-MVP\backend\data\ieee` (train_transaction.csv 683 MB, 590k rows; train_identity.csv 26 MB).

## Audit (2026-11-18)

- Historical orienteers PR-AUC 0.444182 / F1 0.464247 confirmed as `ieee_final_v2.metrics.json`
  (test split, chronological 70/15/15 on TransactionDT; threshold from validation F1; model_version `ieee-cis-xgboost-v2`).
- Baseline features: only 3 V-features (V1..V3), 1e2 rows used via `--max-rows` variants; full-model metrics above.
- Backend integration: `backend/app/ml.py: predict_ieee` loads `MODEL_PATH` artifact (format_version 1), schema check, returns score/threshold/flagged. No IEEE model is sent to production automatically.
- IBM AML agent works in main repo on `feat/ibm-shadow-export`; this worktree must not touch those files.
- No lightgbm/catboost installed; xgboost 3.4.1 available in main venv. Python 3.13.
- Kaggle CLI 2.2.5 authorized; competition `ieee-fraud-detection` entered. Local dataset already present, so no download needed.
- Protocol: chronological holdout; threshold tuned on validation only; test evaluated once per experiment; no final-test tuning.

## Plan

1. v3 trainer: full V/C/D/M features + engineered amount/time/email/freq features, early stopping on validation, extended metrics (P@100/P@1000, FP/10k).
2. Compare vs ieee_final_v2 on identical split protocol. Keep better model as experimental candidate only.
3. Integration: artifact format stays v1-compatible; backend schema check needs no change if features list changes (artifact carries its own feature list).

## Log

- 2026-11-18: Audit done. ieee_final_v2 metrics verified on disk (test PR-AUC 0.444182, F1 0.464247, FP/10k 129.6, threshold 0.8222 from validation). Backend loads model via `IZQAR_MODEL_PATH` (default `backend/models/ieee_model.joblib`, gitignored); missing model -> predict returns None -> API 503 (covered by test).
- Added `backend/scripts/check_ieee_artifact.py` (artifact structure + NaN-row inference check, exit non-zero on failure).
- Added `backend/tests/test_ieee_artifact.py` (3 tests, monkeypatched artifact; all pass; full suite 9 passed; Windows temp-db cleanup PermissionError is a teardown quirk, not a failure).
- Added `backend/scripts/train_ieee_v3.py`: full C/D/V features + engineered (amt_log, amt_cents, dt_hour, dt_day, email_match, v_missing, d_missing, train-fold frequency encodings for card1/card1_card5/card2/addr1/P_emaildomain/DeviceInfo), native NaN handling, early stopping on validation aucpr, extended metrics (P@100/P@1000, FP/10k, FN). Same 70/15/15 chronological split as baseline; threshold from validation F1 only; test evaluated once.
- Smoke run (60k rows, 60 trees) completed OK; full run launched in background.
- 2026-11-18: Full v3 run done (1172 s). Test PR-AUC 0.5074 (+0.0632 vs baseline), F1 0.4915, P@100 1.0, P@1000 0.852, FP/10k 94.0 (vs 129.6), ROC-AUC 0.8987, recall flat −0.001. Artifact check OK. Details: docs/IEEE_V3_RESULTS.md. NOTE: v3 metrics were produced with freq features unused and non-causal freq; fixed in 9517541 and re-evaluated in research mode on Kaggle GPU (see below).
- v3 artifact preserved as reproducible experimental baseline: `models/experiments/ieee_v3_candidate.joblib` (+metrics). NOT promoted to production; default `IZQAR_MODEL_PATH` unchanged.
- 2026-11-18: Stopped orphaned `ieee_v4_candidate` (3000-tree run with pre-fix code; PID 18168/16624 verified via Win32_Process before Stop-Process). No v4 artifacts produced; v3 candidate untouched (sha not required, file size 1,178,070 bytes).
- Bug fix: freq features were dropped from training (`nums` computed from pre-split frame) — v3 used 401 features instead of 407. Fixed; regression tests added.
- Freq features are now strictly causal (chronological prior-occurrence cumcount); no future information even inside train period (v3's value_counts leaked future frequencies within train).
- Evaluation split into research (default; 3 expanding chronological folds on first 85%, tail never evaluated, no artifact written) and final (historical benchmark) modes.
- pytest exits 0 (SQLite engine disposed at exit in conftest). 15 passed.
- Kaggle GPU kernel `hoogle57/izqar-ieee-v4-causal-gpu-cv-v1` pushed (pinned revision 9517541, full features + causal FE, research CV only, arms causal vs none on identical folds). Results pending.
 (1172 s). Test PR-AUC 0.5074 (+0.0632 vs baseline), F1 0.4915, P@100 1.0, P@1000 0.852, FP/10k 94.0 (vs 129.6), ROC-AUC 0.8987, recall flat −0.001. Artifact check OK (401 features). Details: docs/IEEE_V3_RESULTS.md.
- Follow-up run with --estimators 3000 (early stopping still enabled) launched 2026-11-18 to test whether the 1500-tree cap was binding.
