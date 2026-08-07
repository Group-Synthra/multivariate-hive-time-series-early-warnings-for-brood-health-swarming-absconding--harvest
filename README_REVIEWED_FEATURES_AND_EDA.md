# Reviewed Feature Engineering and EDA Milestone

This package is the next step after the reviewed target dataset has been built.

## Why chronological CV remains empty

The official training split contains nine reviewed events, but eight occur at
almost the same time. A valid purged expanding-window fold therefore cannot be
created. Keep the official split unchanged.

Evaluation design:

1. Primary: official validation split with two reviewed events.
2. Final test: one-event case study.
3. Secondary: leave-one-positive-hive-out sensitivity folds using only the
   official training split.

## What the feature builder does

- Uses only current and past observations.
- Resets all rolling history after a non-hourly gap.
- Requires 168 hours of contiguous history.
- Excludes every target/event/label column from model features.
- Produces weight, temperature, humidity, CO2, time and quality features.
- Saves a feature manifest and audit report.

## Installation

Copy the package's `backend` folder into the repository root.

Append the contents of:

`backend/config/harvesting_reviewed_features_section.yaml`

to:

`backend/config/harvesting.yaml`

Then remove the snippet file.

## Commands

From `backend`:

```powershell
ruff check . --fix
ruff check .
pytest -v

python scripts/build_reviewed_harvest_features.py
python scripts/run_reviewed_harvest_feature_eda.py
python scripts/build_grouped_hive_validation.py
```

## Outputs

Feature engineering:

- `data/processed/harvest_reviewed_feature_dataset.parquet`
- `artifacts/reports/harvesting/reviewed/features/feature_manifest.csv`
- `artifacts/reports/harvesting/reviewed/features/feature_audit.json`

Reviewed feature EDA:

- `artifacts/reports/harvesting/reviewed/feature_eda/event_feature_samples.csv`
- `artifacts/reports/harvesting/reviewed/feature_eda/matched_control_samples.csv`
- `artifacts/reports/harvesting/reviewed/feature_eda/sample_coverage.csv`
- `artifacts/reports/harvesting/reviewed/feature_eda/lead_feature_comparison.csv`
- `artifacts/reports/harvesting/reviewed/feature_eda/top_features_by_lead.csv`
- `artifacts/reports/harvesting/reviewed/feature_eda/reviewed_feature_eda_audit.json`
- one feature-difference figure for each lead time

Grouped sensitivity folds:

- `artifacts/reports/harvesting/reviewed/grouped_hive_folds.csv`
- `artifacts/reports/harvesting/reviewed/grouped_hive_summary.json`

## Stop point

Review the feature audit, EDA audit, top features and grouped fold summary before
training models. Do not use the empty chronological `cv_folds.csv` as a model
input.
