# IZQAR: IBM AML causal-history shadow workflow

This workflow is intentionally **validation-only**. It exports the trained
preprocessing and XGBoost weights from a *new run* of `causal_history_v1`
without touching the previously viewed final test. It cannot recover weights
from earlier **report-only** Kaggle ZIPs; those ZIPs contain only metrics.

## What the workflow does

1. The IBM trainer runs in `--mode experiment`, using chronological
   train and validation periods for learning and model selection. The full
   timestamp column is scanned to establish split boundaries; the sealed
   test labels are not used for fitting, threshold tuning or scoring.
2. With `--shadow-bundle`, it also writes one ZIP containing the **fitted**
   preprocessor, XGBoost JSON, validation-only metadata and a SHA-256 manifest.
3. Before completing export, it round-trips the fitted model on a single
   **synthetic schema row** and requires near-identical scores.
4. Offline ZIP verification checks member names, size limits, CRC, SHA-256,
   approved historical feature schema, and strict `run_mode=experiment`.
5. Installation is permitted only under a directory named `experiments`.
   Candidate files are not an API promotion. The generic specialist API
   loader rejects `run_mode=experiment` manifests before unpickling.

**Important:** a causal-history model needs history features computed from
*strictly earlier* transaction timestamps. A request with a few static
values cannot reproduce these features. The synthetic smoke only checks the
serialization and prediction contract, not detection quality. Do not expose
this candidate through `/api/v1/ibm-aml/predict` or claim real-bank readiness.

## Safe Kaggle run

Use `notebooks/ibm_causal_shadow_kaggle.ipynb`, pinning the exact Git SHA of
`feat/ibm-shadow-export`, in a fresh Kaggle notebook. Attach IBM HI-Small using an allowed Kaggle Input. Do
not replace the old saved report ZIP or the active IBM artifact. The
following is one **new experiment**, not a promotion or final evaluation:

```bash
python backend/scripts/train_ibm_aml.py \
  --data /kaggle/input/your-ibm-dataset/HI-Small_Trans.csv \
  --mode experiment --feature-set causal_history_v1 \
  --full-train --estimators 180 \
  --max-false-alerts-per-10000 100 \
  --output /kaggle/working/ibm_causal_shadow_v1.joblib \
  --shadow-bundle /kaggle/working/ibm_causal_shadow_v1.zip
```

`--full-train` uses the **full early training period**, not the final holdout.
Omit it to test a smaller capped training period. A new `--early-stopping-rounds`
value may be chosen only through the documented validation protocol. To
reproduce an earlier benchmark exactly, use its **recorded and pinned**
hyperparameters, code revision and input dataset, then compare only the
validation metrics. These instructions do not claim exact reproducibility
without that comparison.

Record the Kaggle commit SHA, kernel/version, and the **new ZIP SHA-256**.
The archive is potentially large. Never commit it, the original CSV,
credentials, or extracted `.joblib` to Git.

## Verify and install on Windows

Check locally in the environment that will run the preprocessor. The
`verify` command only parses ZIP/JSON and does not execute pickle contents.

```powershell
cd C:\Users\a\Documents\IZQAR-MVP\backend
$zip = 'F:\download\ibm_causal_shadow_v1.zip'
$sha = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant()
.\.venv\Scripts\python.exe scripts\ibm_shadow_bundle.py verify --bundle $zip --expected-sha256 $sha
```

**Caution:** self-calculated SHA-256 verifies that bytes are unchanged *since
that calculation*. For source provenance, compare the value with the hash
recorded separately in the successful Kaggle run. A hash is not a signature.

To install this ZIP **only for offline tests**, use a fresh folder under
`models/experiments`:

```powershell
.\.venv\Scripts\python.exe scripts\ibm_shadow_bundle.py install `
  --bundle $zip --expected-sha256 $sha `
  --destination models\experiments\ibm_causal_shadow_v1 `
  --trusted-smoke
```

`--trusted-smoke` explicitly loads the preprocessor via `joblib`, which uses
pickle. Only run it for an authenticated, trusted artifact with the
expected hash. It makes one synthetic prediction and **does not** measure
accuracy or compare against bank data.

The Model Registry can list the portable candidate with `lifecycle=candidate`.
Its `state=installed` only means components exist. It is **not** a serving
model, regardless of that state. Even if the path is accidentally configured
as active, the hardened Model Registry reports it as a candidate with an
explicit configuration error; the API refuses to load it.


## New experimental branch: bank-scoped causal history v2

The original `causal_history_v1` indexes prior counts and amounts by the
account string alone. In systems where an account identifier is unique only
within its bank, this can merge histories from two distinct bank/account
pairs. **This is a potential model-design defect, not a measured defect in
IBM HI-Small** until bank/account collisions are audited on the dataset.

`causal_history_bank_v2` is a separate **validation-only** feature-set option.
It keys state by the tuple `(From Bank, Account)` for outgoing transactions and
`(To Bank, Account.1)` for incoming transactions. The bank/account tuple never
appears as an ML feature. Equal timestamps remain a batch, preserving
causality; the six numeric feature names remain the same but their **meaning
changes**, so the v2 model receives a distinct version identifier. Existing
v1 metrics and v1 artifacts must not be relabeled as v2.

Use `notebooks/ibm_causal_bank_v2_kaggle.ipynb` for the experiment. It starts
with `EXPECTED_REVISION` unset and refuses to train until the *exact published*
commit SHA is pinned. Only a fresh Kaggle run can supply v2 validation scores
or a v2 shadow ZIP; **no new v2 performance claims are made by this patch**.

Both v1 and bank-scoped v2 use a bounded history map. Once the account cap is
reached, new accounts cold-start rather than being tracked. Record the cap
when comparing candidates, because it may affect the resulting features.

## Quantify bank/account collisions without reading AML labels

The v2 design addresses a potential ambiguity. Whether it matters in the
**actual IBM HI-Small early train period** is not established by synthetic
unit tests. `audit_ibm_account_collisions.py` scans only the five identity and
timestamp columns, using disk-backed SQLite. It deliberately does NOT read
`Is Laundering`, write raw identifiers into the result or calculate any
model metric. Use the **same validation_start** recorded in the original
training report; do not look at test labels.

```powershell
cd C:\Users\a\Documents\IZQAR-MVP\backend
.\.venv\Scripts\python.exe scripts\audit_ibm_account_collisions.py `
  --data 'PATH_TO_HI-Small_Trans.csv' `
  --train-before 'TIMESTAMP_FROM_split_boundaries.validation_start' `
  --output models\experiments\ibm_account_collision_audit.json
```

Only aggregate counts are saved: distinct bank/account pairs, account strings
seen in multiple banks, and the share of such strings. An optional
`--max-train-rows` is only a partial diagnostic and must be labelled as such.
The scan uses temporary SQLite and can be I/O intensive on the full CSV.

## Compare candidates on an identical validation population

The read-only comparison tool refuses to compare mismatched source dataset
SHA-256, validation boundaries, population counts or false-alert budgets.
It never reads pickle files or test metrics and never promotes a model.

```powershell
cd C:\Users\a\Documents\IZQAR-MVP\backend
.\.venv\Scripts\python.exe scripts\compare_ibm_validation.py `
  --baseline models\ibm_aml_hi_small_v1_portable.metrics.json `
  --candidate models\experiments\ibm_causal_shadow_v1\ibm_causal_shadow_v1.metrics.json `
  --output models\experiments\ibm_causal_shadow_v1\comparison.json
```

The baseline report path is illustrative. Substitute your actual installed
IBM report. If the tool rejects the comparison, **do not bypass the check**:
check the source CSV hash, chronological split, `cases`, and FP budget.

## Verify installed files without deserializing

The original `verify` command checks the ZIP itself, while
`verify-installed` rehashes each installed component against its manifest
before any optional trusted pickle smoke:

```powershell
.\.venv\Scripts\python.exe scripts\ibm_shadow_bundle.py verify-installed `
  --destination models\experiments\ibm_causal_shadow_v1 `
  --stem ibm_causal_shadow_v1
```

The `--trusted-smoke` option also executes this installed-file integrity
check **before** calling `joblib.load`. A local manifest is not a cryptographic
signature; keep the separately recorded Kaggle SHA-256 and verify your source.
This is not a banking security certification.

## Stop conditions

- No clean source revision or unknown Kaggle dataset: **do not export**.
- Validation metrics missing, forbidden features, final-test metrics,
  mismatched SHA-256, unexpected ZIP entries, or overwrite: **reject**.
- Incompatible local scikit-learn version: stop before trusted smoke. Use a
  compatible environment rather than silently suppressing warnings.
- Repeated test-set tuning or automatic promotion: **not allowed**.
- New candidate scores not as good as documented comparison on the same
  validation window: keep the bundle for research only, do not change
  production/demo settings.

## What remains to be done

This work provides a safe export/import and schema smoke. It does **not**
implement a stateful online account-history serving service, authenticated
operator approvals, external banking integration, or a new final-test
result. A future analyst-facing shadow inference path would need a durable
history store, canonical timestamp grouping, deduplication, replay
protection, and independent provenance review.
