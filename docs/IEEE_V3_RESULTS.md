# IEEE-CIS v3 Candidate — Results (2026-11-18)

Protocol (identical to `ieee_final_v2`): 70/15/15 chronological split by `TransactionDT`,
threshold tuned on **validation** by max-F1, held-out test evaluated **once** per run.

## Headline (held-out test)

| Metric | ieee_final_v2 (baseline) | ieee_v3_candidate | Δ |
|---|---|---|---|
| PR-AUC | 0.444182 | 0.507407 | +0.0632 |
| ROC-AUC | 0.875369 | 0.898684 | +0.0233 |
| Precision | 0.526988 | 0.605026 | +0.0780 |
| Recall | 0.414856 | 0.413883 | −0.0010 |
| F1 | 0.464247 | 0.491525 | +0.0273 |
| Precision@100 | n/a (not recorded) | 1.000 | — |
| Precision@1000 | n/a (not recorded) | 0.852 | — |
| False alerts / 10k | 129.6 | 94.0 | −35.6 |
| False negatives | 1804 | 1807 | +3 |

Validation: PR-AUC 0.5659 vs baseline 0.4552; P@100 0.99; FP/10k 70.3.

## What changed (train_ieee_v3.py)

- Full numeric feature set: all C1–C14, D1–D15, V1–V339 (was: V1–V3 only).
- Engineered: `amt_log`, `amt_cents`, `dt_hour`, `dt_day`, `email_match`,
  `v_missing`, `d_missing`.
- Train-fold frequency encodings (no leakage): card1, card1×card5, card2, addr1,
  P_emaildomain, DeviceInfo.
- Native XGBoost NaN handling for numerics; OHE (min_frequency=100, max 60) for categoricals;
  early stopping on validation aucpr (best_iteration hit the 1500-tree cap → follow-up run with 3000).
- Artifact format unchanged (`format_version: 1`) → compatible with `app.ml.predict_ieee`
  and `IZQAR_MODEL_PATH`.

## Artifact

`models/experiments/ieee_v3_candidate.joblib` + `.metrics.json` (gitignored, regenerate via
`backend/scripts/train_ieee_v3.py`). Structure verified with `backend/scripts/check_ieee_artifact.py`.

## Status / caveats

- Experimental candidate only; production default (`IZQAR_MODEL_PATH` → `backend/models/ieee_model.joblib`) unchanged.
- Test scores are a single chronological holdout, not repeated CV; do not treat as final proof.
- Scores are not calibrated probabilities.
- Recall essentially flat (−0.001): improvement is in ranking/precision, not fraud capture rate.
