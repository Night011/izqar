# IEEE-CIS v4 — Research-mode CV results (Kaggle GPU, 2026-11-18)

Revision `5a8b8a41` (tail physically excluded); v1 kernel on `9517541` with identical
absolute fold boundaries. Final 15% tail never loaded for evaluation. Three arms,
same folds, same seed, 1500 trees, early stopping (60) on fold validation aucpr:

| Arm | feat | Fold1 (valid<0.55) PR-AUC | Fold2 (valid<0.70) PR-AUC | Fold3 (valid<0.85) PR-AUC | Mean PR-AUC | Mean F1 | Mean P@100 | Mean P@1000 | Mean FP/10k |
|---|---|---|---|---|---|---|---|---|---|
| causal (raw cumcount) | 407 | 0.579366 | 0.552038 | 0.548628 | 0.5600 | 0.5458 | 0.983 | 0.906 | 96.7 |
| none (no freq) | 401 | 0.585849 | 0.570954 | 0.569276 | 0.5754 | 0.5548 | 0.997 | 0.911 | 88.8 |
| causal_log (log1p cumcount) | 407 | 0.587796 | 0.558844 | 0.524176 | 0.5569 | 0.5395 | 0.993 | 0.905 | 77.5 |

Per-fold detail (PR-AUC / F1 / P@100 / P@1000 / FP per 10k):

- causal: (0.5794, 0.566, 0.990, 0.938, 83.9) (0.5520, 0.533, 0.970, 0.906, 96.3) (0.5486, 0.539, 0.990, 0.875, 110.0)
- none:   (0.5858, 0.574, 0.990, 0.943, 61.4) (0.5710, 0.544, 1.000, 0.901, 98.0) (0.5693, 0.546, 1.000, 0.890, 106.9)
- causal_log: (0.5878, 0.573, 0.990, 0.944, 52.5) (0.5588, 0.537, 1.000, 0.910, 83.3) (0.5242, 0.509, 0.990, 0.862, 96.7)

## Findings

1. Frequency features (both raw and log-scaled, strictly causal) do NOT improve
   cross-period ranking on identical folds; causal hurts mean PR-AUC by ~0.015 and
   causal_log by ~0.019. The effect is consistent in 2 of 3 folds and in the mean.
2. Correct v4 configuration = v3-fixed configuration (full C/D/V + engineered
   non-freq features). `ieee_v3_candidate.joblib` already carries it (401 features).
3. No retraining performed: the v4 candidate configuration equals the validated
   v3 artifact configuration; re-running would be a redundant experiment.
4. Causal freq implementation retained (`--freq causal|causal_log`) for future
   validation of history-bearing deployments; inference supports it via
   `app/ieee_freq.py` (batch exact / single-row freq_state).
5. Historical benchmark (previously used test): PR-AUC 0.5074, F1 0.4915, P@100 1.0,
   FP/10k 94.0 — NOT an independent evaluation; for reference only.

## Caveats

- Folds are single chronological expanding windows; fold differences of ~0.006-0.018
  PR-AUC may partly be noise, but the direction is consistent across the mean and FP budgets.
- Scores uncalibrated; candidate not promoted to production.
