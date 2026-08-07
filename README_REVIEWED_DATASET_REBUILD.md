# Reviewed Harvest Dataset Rebuild

This milestone rebuilds the 72-hour forecasting dataset from the manually
reviewed probable-harvest events.

## Why a new dataset is required

The original generated markers included 27 delayed timestamps. The reviewed
event table corrects or excludes those events. The old model dataset must not
be reused for final training.

The rebuilt dataset also excludes the event timestamp and the following
24-hour recovery period. Without this exclusion, the classifier could learn
large post-removal weight artefacts as ordinary negative examples.

## Install

Copy the package's `backend` folder into the repository root.

Append the contents of:

`backend/config/harvesting_reviewed_section.yaml`

to the existing:

`backend/config/harvesting.yaml`

Then remove the snippet file.

## Run

From `backend`:

```powershell
ruff check . --fix
ruff check .
pytest -v

python scripts/run_reviewed_harvest_dataset.py
python scripts/build_reviewed_harvest_cv_folds.py
```

## Outputs

- `data/processed/harvest_reviewed_72h_dataset.parquet`
- `artifacts/reports/harvesting/reviewed/target_audit.json`
- `artifacts/reports/harvesting/reviewed/target_balance_by_split.csv`
- `artifacts/reports/harvesting/reviewed/cv_folds.csv`
- `artifacts/reports/harvesting/reviewed/cv_summary.json`

## Stop point

Inspect `target_audit.json` and `cv_summary.json` before feature engineering.
With only 12 reviewed events, an official validation or test split may contain
zero usable events. The evaluation design must be based on the actual rebuilt
counts rather than assumed counts.
