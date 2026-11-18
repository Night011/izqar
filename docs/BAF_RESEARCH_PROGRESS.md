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

## Stage 2 — error analysis (v1, train months [0..4], valid month 5)

- Metrics (00th-pct threshold): PR-AUC 0.167 (fold-5), F1 0.246 at 73 FP/10k (99th pct).
- Dominant signal: housing_status_BA (~0.49 mean |SHAP|, 0.32 importance) — 4.1% fraud rate segment with 770/873 FPs; device_os=windows (0.42), phone_home_valid, has_other_cards, keep_alive_session, name_email_similarity, income, prev_address_months_count missing indicator.
- FN concentration: employment_status CA/CB/CF (FN share 0.75/0.95/1.0), housing_status BC/BE (0.93/1.0), device_os linux/macintosh (0.94/0.76). Frauds hide in these segments; FPs concentrate in BA/windows/free-email segments — inherent imbalance trade-off.
- Stability: same top features across months; name_email_similarity weaker but stable mid-rank.
- Not promoted: v2 engineered features (no fold gain). Velocity aggregates excluded from candidates (Stage 4).

## Stage 3 — in flight

- Kaggle GPU kernel `hoogle57/izqar-baf-algorithms-gpu-v1` (rev 97ebd1b): XGBoost vs LightGBM vs CatBoost × v1/v2 on identical folds. Status: QUEUED.
- Controlled local arm: v1 with max_depth=6 (only change vs v1) — running (PID 5244).

## Log

- 2026-11-18: Audit done; Base.csv downloaded locally (1,000,000 rows; 1.103% fraud; months 0–7; train [0–5], valid 6, test 7 — test never used for tuning).
- Research folds (train region only, expanding: valid months 3/4/5; threshold from fold valid, FP/10k budget 100; 400 trees, early stop 40):

| Config | Fold3 (0.141) PR-AUC | Fold4 | Fold5 | Mean PR-AUC | Mean F1 | Mean FP/10k | P@100 (folds) | P@1000 (folds) |
|---|---|---|---|---|---|---|---|---|
| v1 (20 feat, XGB 180-config) | 0.141364 (rows 397k) | 0.160951 (548k) | 0.167087 (676k) | 0.156467 | 0.2266 | 99.6 | 0.46/0.50/0.50 | 0.230/0.257/0.291 |
| v2 (+12 engineered) | 0.138503 | 0.159649 | 0.166637 | 0.154930 | 0.2298 | 99.1 | 0.49/0.46/0.48 | 0.225/0.262/0.282 |
| velocity (v2 + 7 velocity aggs) | 0.154405 | 0.177726 | 0.189083 | **0.173738** | 0.2426 | 99.5 | 0.55/0.56/0.59 | 0.235/0.264/0.310 |

- v2 engineered features: NO measurable gain (mean PR-AUC −0.0015). Velocity aggregates: +0.0173 mean PR-AUC, P@100 +0.10 — meaningful headroom, but NOT deployable without a verified streaming aggregation pipeline (application-time availability unverified). Decision: candidate keeps v1 feature contract + tuning; velocity documented as blocked enhancement.
- Kaggle GPU kernel `hoogle57/izqar-baf-algorithms-gpu-v1` pushed (pinned 97ebd1b): XGBoost vs LightGBM vs CatBoost × v1/v2 on identical folds. Pending.
