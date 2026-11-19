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
- Kaggle GPU kernel v1 (hoogle57/izqar-ieee-v4-causal-gpu-cv-v1): arms causal vs none on identical folds. causal mean PR-AUC 0.5600 vs none 0.5754 (none better on all 3 folds). Kernel COMPLETE; benign cuda/cpu UserWarnings only. Metrics committed (kaggle/outputs/).
- Kaggle kernel v2 (causal_log arm, 5a8b8a4): mean PR-AUC 0.5569 — no gain. Frequency features rejected by the evidence; v4 candidate = v3-fixed configuration (401 features). Full report: docs/IEEE_V4_CV_RESULTS.md.
- Research mode now physically excludes the final 15% tail before any processing (verified: excluded_tail_rows=88581 on full data); folds relative to truncated pool.
- Freq reproduction at inference implemented + verified (app/ieee_freq.py, score_ieee_batch.py, predict_ieee freq_state fallback, check_ieee_artifact freq_state guard, 144 tests passed).
- CI root cause: check-run annotation on ALL runs (every branch incl. main): "The job was not started because your account is locked due to a billing issue." CI does not execute at all; local pytest is the verification path.
- IBM branch merged into feat/ieee-cis-research (cc48db2): conftest now byte-identical to IBM's (SQLite teardown fixture, pytest exits 0), .gitignore/main.py/ml.py unions contain both agents' changes (ieee-candidate endpoint + investigations/model-registry). 140 tests passed pre-freq-machinery, 144 after.
- 2026-11-18: PR #1 MERGED into main (merge commit b6c8c5b). Working tree == origin/main (empty diff). 144 tests passed on merged state. CI cannot run: billing lock (see GitHub Actions annotation). No deployment procedure exists in repo (no deploy workflow/scripts/infra config) — deployment not performed; local demo path documented in README unchanged. Rollback: main previous state 0d2e9ea; experiment models remain local (gitignored), previous production model paths untouched.

## v5-v7: diversity levers on identical folds (2026-11-19)

Research mode, same three expanding folds, final 15% tail physically excluded.

| run | arm | mean PR-AUC | recall@100 FP/10k | verdict |
|---|---|---|---|---|
| v5 | `none` + seeds 42/43/44, rank-average | 0.5757 | — | seed ensemble: +0.0003, nothing |
| v6 | `none` alone | 0.5765 | — | reference |
| v6 | `causal` alone | 0.5569 | — | weaker by 0.0196 |
| v7 | `none` + `causal`, rank-average | 0.5720 | 0.474 | heterogeneous blend: **worse** by 0.0045 |

v6 could not answer the question at first: `--scores-dir` was repeated instead of
carrying both values, so it blended one arm and reported that arm's metrics. Fixed
in `ieee_blend_scores.py` (`nargs='+'`); v7 is the corrected run.

Conclusion: IEEE-CIS sits on the same plateau as BAF. Diversity does not pay
because the `causal` arm's errors are a subset of `none`'s, not complementary.
The remaining real gap is promotional, not modelling: the production artifact
measures test PR-AUC 0.4442 while the v3-fixed configuration measures 0.5074 on
the same historical test and 0.5765 in research CV, and it was never exported as
a candidate.
