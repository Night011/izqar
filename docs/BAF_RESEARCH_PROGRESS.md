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
- max_depth=6 REJECTED: mean PR-AUC 0.1517 (0.1361/0.1551/0.1639) vs v1 0.1565.
- lr=0.03, 800 trees, early stop 60 — REJECTED: mean PR-AUC 0.1558 (0.1404/0.1593/0.1678) vs v1 0.1565. baseline plateau persists.
- Threshold drift across months: 0.939 (m3) → 0.935 (m4) → 0.913 (m5) with fraud-rate 0.92%→1.18%; fixed threshold not stable — budget-aware per-month recalibration is appropriate.
- Kaggle kernel QUEUED ~50+ min without diagnostics; local fallback: isolated venv (temp) with lightgbm+catboost for identical folds.

## Stage 3 experiment ledger (identical folds, months 3/4/5, 1500-400 iters)

| Config | Mean PR-AUC | Folds | Verdict |
|---|---|---|---|
| xgboost v1 (280-config, depth 4) | 0.1554/0.1565 | 0.140/0.158/0.168 | reference best so far |
| xgboost v2 | 0.1538/0.1549 | 0.139/0.156/0.166 | rejected (-0.002) |
| xgboost max_depth=6 | 0.1517 | 0.136/0.155/0.164 | rejected |
| xgboost lr=0.03, 800 trees | 0.1558 | 0.140/0.159/0.168 | rejected (~noise) |
| catboost (OHE) v1 / v2 | 0.1572 / 0.1578 | ~0.144/0.161/0.167 | parity with XGB (+0.002) |
| catboost native (no OHE) v1 / v2 | 0.1556 / 0.1562 | 0.143/0.162/0.167 | parity (no gain vs OHE) |
| lightgbm v1/v2 (SPW or is_unbalance) | 0.052 | 0.048/0.052/0.057 | BROKEN arm — early-stops at iter 1; even tuned (is_unbalance, depth=-1, leaves=31) PR-AUC 0.053. Dropped, documented as LightGBM failing on this workload/config, not on LightGBM per se |
| velocity headroom (v2 + 7 aggs) | 0.1737 | 0.154/0.178/0.189 | NOT eligible (Stage 4) |
| XGB+CatBoost rank-average ensemble | 0.1579 (v1) / 0.1571 (v2) | 0.143/0.163/0.167 | parity — NOT adopted (no gain ≥ error) |
| **Decision**: no admissible configuration beats XGBoost-v1 on internal folds. Ceiling ≈ 0.156–0.158 mean PR-AUC across the best three arms. Baseline model stands. |

- 2026-11-18 (late): final verdict recorded. BAF experimental endpoint `/api/v1/model/baf-candidate` added (read-only; metrics of research folds; 404 when absent). 147 tests passed. Ensemble NOT adopted (P@100 flat 0.5 vs XGB 0.46/0.50/0.50, within noise).
- Recommendation: ship nothing new; document ceiling. Any further gains would need the verified streaming velocity pipeline (excluded by Stage 4) or stronger raw signals.

## Stage 7 — corrected evaluation protocol (validity fix)

Problem: `threshold_analysis(fold_valid.fraud_bool, scores, 100)` chose the operating point on the same fold it reported, so FP/10k ≈ 99.6 was guaranteed by construction. Same fold also drove early stopping.

Protocol (`backend/scripts/baf_protocol.py`, `--protocol calibrated`): for external month *m*, fit on months < *m*−1, choose **both** the round count and the threshold on month *m*−1, score month *m* once with both frozen. `--protocol infold` reproduces the old layout and is labelled optimistic in the artifact. Months 6/7 never enter selection. Regression tests: `backend/tests/test_baf_protocol.py` (8 tests; **155 backend tests pass**; CI remains blocked by the GitHub Actions billing lock).

XGBoost v1, calibrated (`docs/baf_calibrated_v1.metrics.json`):

| External fold | Train / cal | PR-AUC | ROC | P@100 | P@1000 | Recall | F1 | FP/10k (frozen) | FP/10k on cal | Rounds |
|---|---|---|---|---|---|---|---|---|---|---|
| m3 | m0–1 / m2 | 0.137190 | 0.9043 | 0.45 | 0.226 | 0.3621 | 0.2016 | **205.7** | 99.87 | 304 |
| m4 | m0–2 / m3 | 0.156176 | 0.8998 | 0.46 | 0.260 | 0.2417 | 0.2190 | **109.9** | 99.18 | 264 |
| m5 | m0–3 / m4 | 0.166224 | 0.8853 | 0.48 | 0.298 | 0.1906 | 0.2350 | 51.0 | 99.77 | 228 |
| mean | | 0.153197 | 0.8965 | 0.463 | 0.261 | 0.2648 | 0.2185 | **122.2** | 99.6 | |

Comparison with the in-fold protocol (old, optimistic): mean PR-AUC 0.1565, F1 0.2266, FP/10k 99.6.

- Ranking quality is unchanged: P@100 0.45/0.46/0.48 and P@1000 0.226/0.260/0.298 match the old 0.46/0.50/0.50 and 0.230/0.257/0.291 within noise. The old optimism was **entirely in the operating point**, not in the model.
- With a genuinely pre-registered threshold the 100 FP/10k budget is breached on 2 of 3 folds (205.7, 109.9) while calibration itself is always ≈99.7 — the score distribution shifts between months (fraud rate 0.92%→1.18%).
- Post-hoc recall at 50/75/100 FP/10k on the external folds (diagnostic; recalibrating per period needs labels): 0.159/0.197/0.233 · 0.168/0.196/0.236 · 0.190/0.238/0.266.
- Fold m3 now trains on 260k rows instead of 397k (the extra month became calibration), so its PR-AUC is slightly lower; this is the honest cost of the protocol.

## Stage 7b — algorithm comparison under the corrected protocol (run B)

`docs/baf_calibrated_algos_v1.metrics.json` — 6 arms × 3 calibrated folds, identical periods, identical selection rule (rounds and threshold on the calibration month only), v1 features.

| Arm | PR-AUC (f3/f4/f5) | Mean PR-AUC | Mean F1 | FP/10k frozen (f3/f4/f5) | Recall@100 post-hoc | P@1000 (f3/f4/f5) | Rounds | Fit s (f5) |
|---|---|---|---|---|---|---|---|---|
| catboost (OHE) | .1392/.1623/.1657 | **0.1558** | 0.2186 | 212.9/106.9/52.5 | .239/.233/.262 | .223/.253/.291 | 350/275/350 | 52 |
| ensemble xgb+cb_native | .1399/.1598/.1670 | **0.1556** | **0.2258** | **98.2/95.6/96.1** | .237/.233/.264 | .230/.260/.297 | 300/400 · 250/375 · 200/400 | 180 |
| catboost_native | .1394/.1594/.1648 | 0.1545 | 0.2213 | 210.7/104.8/50.5 | .236/.234/.262 | .230/.253/.288 | 400/375/400 | 177 |
| lightgbm_reg | .1365/.1582/.1677 | 0.1541 | 0.2195 | 202.5/107.4/49.6 | .242/.238/.261 | .222/.261/.289 | 750/800/800 | 29 |
| xgboost v1 | .1372/.1565/.1652 | 0.1529 | 0.2199 | 204.3/109.9/51.4 | .233/.233/.264 | .226/.262/.297 | 300/250/200 | 33 |
| lightgbm (unweighted) | .1312/.1568/.1608 | 0.1496 | 0.2163 | 207.2/108.0/51.8 | .241/.238/.265 | .218/.251/.286 | 250/200/150 | 20 |

Conclusions:

- **No arm separates on ranking.** PR-AUC spans 0.1496–0.1558 with per-fold spread ≈0.01; P@1000 and post-hoc recall@100 are within ±0.01 for all six. Algorithm choice is not the bottleneck; the v1 feature signal is. This is a measured plateau over 6 configurations, 3 folds and one dataset variant — not a proven ceiling.
- **The old optimism was the operating point, not the model.** Five arms breach 100 FP/10k on folds m3/m4 (200–213 / 105–110) while their calibration months sit at ≈99.7. A probability threshold fitted on one month does not transport to the next (fraud rate 0.92%→1.18%).
- **Ensemble is the only arm inside the budget on all folds (98.2/95.6/96.1).** Cause: rank averaging makes the score distribution uniform, so the calibration threshold is effectively a within-batch quantile. That is *not* available for single-request scoring without a rolling reference window — so its operating-point advantage is batch-dependent and not adopted.
- **LightGBM is not broken.** Root cause of the old 0.052: `eval_metric='aucpr'` is an XGBoost name, LightGBM ignored it and monitored weighted `binary_logloss`, which rises by construction under `scale_pos_weight`; early stopping kept round 1 and LightGBM discarded every later tree (`current_iteration() == 1`). With explicit round selection on calibration PR-AUC, LightGBM reaches 0.1496 (unweighted) and 0.1541 (regularised), i.e. parity.

Verdict: research stage BAF closed on the current feature contract. No configuration beats the baseline beyond noise; nothing is promoted to production, and the production artifact is untouched. Remaining headroom needs new admissible signals (a verified streaming velocity pipeline), not more tuning.

Reproduction (local, `backend/data/baf/Base.csv`, venv with xgboost 3.4.1 / lightgbm 4.7.0 / catboost 1.2.10 / sklearn 1.9.1):

```powershell
python backend/scripts/train_baf.py --data backend/data/baf/Base.csv --output models/experiments/baf_calibrated_v1.metrics.json --mode research --protocol calibrated --feature-set v1 --estimators 400 --early-stopping-rounds 40
python backend/scripts/research_baf_algorithms.py --data backend/data/baf/Base.csv --output models/experiments/baf_calibrated_algos_v1.metrics.json --algos xgboost catboost catboost_native lightgbm lightgbm_reg ensemble --feature-sets v1 --max-rounds 400 --device cpu
```

Each fold report carries train/calibration/external months, row counts, per-period row-index sha256, best iteration, fit seconds, selected threshold, calibration sweep and full confusion matrix.

## Stage 9 — velocity admissibility, fixed-capacity operating point, pre-deployment refit

Three questions were still open after Stage 8. All three are now measured on the same calibrated folds (train < calibration < external, threshold/rounds frozen; `docs/baf_policy_velocity_v1.metrics.json`).

**1. Are the seven blocked "velocity" columns admissible?** Yes in principle. The official BAF datasheet (`feedzai/bank-account-fraud/documents/datasheet.pdf`, p.2) defines them as trailing-window counts of *prior applications*: `zip_count_4w` = applications in the same ZIP in the last 4 weeks; `velocity_6h`/`24h`/`4w` = average applications per hour over the last 6h/24h/4w; `bank_branch_count_8w`; `date_of_birth_distinct_emails_4w`; `device_distinct_emails_8w`. They are pre-decision business aggregates, not outcomes, so they carry no label leakage. What is still missing is not permission but plumbing: IZQAR must compute the same seven counters over its own application stream (7 rolling-window aggregations keyed by ZIP / branch / DOB / device).

**2. Does that headroom survive the honest protocol?** Yes, and it is the only arm that does.

| Arm | Mean PR-AUC | PR-AUC m3/m4/m5 | Recall @capacity m3/m4/m5 | FP/10k @capacity |
|---|---|---|---|---|
| xgboost v1 | 0.1529 | 0.1372/0.1565/0.1652 | 0.229/0.231/0.260 | 98/96/96 |
| catboost v1 | 0.1558 | 0.1392/0.1623/0.1657 | 0.237/0.227/0.259 | 98/96/96 |
| **xgboost v1v** | **0.1708** (+0.0179) | 0.1501/0.1752/0.1870 | 0.251/0.253/0.279 (+0.022/+0.022/+0.019) | 97/94/95 |
| **catboost v1v** | **0.1732** (+0.0174) | 0.1573/0.1731/0.1891 | 0.254/0.247/0.279 | 98/95/97 |

The gain holds on every fold (+0.011…+0.023) and is ~+10% relative recall at the same false-alert budget. CatBoost's +0.002 lead over XGBoost is inside fold noise and costs 2–3× fit time; XGBoost keeps the tie-break. The earlier 0.1737 estimate (optimistic in-fold threshold) was not inflated by the feature class — it reproduces at 0.1708 under the corrected protocol.

**3. Can the 100 FP/10k budget be met without labels?** Yes, by freezing an alert *rate* instead of a probability. The threshold chosen on the calibration month is translated into the share of that batch it flags (~1.19–1.26%), and the same top share is flagged on the external month. This uses only the score distribution, never external labels, and is the operational equivalent of ranking today's applications and reviewing the top q%. Measured: FP/10k 94–98 on all folds (a frozen probability threshold gave 51–233), and recall lands within **0.0007–0.0069** of the oracle threshold that *is* allowed to fit the external labels at the same budget. On m5 it also gains recall (0.260 vs 0.192) because it spends the budget instead of under-alerting. Cost: serving needs a rolling window of recent scores, since the API scores one request at a time today.

**4. Pre-deployment refit.** Refitting on train+calibration with the round count frozen (calibration then contributes rows, not choices) is worth +0.0017 (XGB) / +0.0019 (Cat) mean PR-AUC, largest on the data-starved m3 fold (+0.0028/+0.0020). Free improvement, no new features, no external data.

Decision: nothing is promoted to production. `v1v` becomes the documented best candidate and the reference for future work, conditional on the seven streaming counters; the capacity policy becomes the recommended serving rule, conditional on a rolling score window. The production artifact (`baf_base_v1_portable`) and its fixed-threshold contract are untouched and remain the rollback point.

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
