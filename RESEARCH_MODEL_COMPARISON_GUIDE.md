# Research-Grade Four-Model Comparison

This milestone keeps the four models originally proposed:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM

The comparison is made more defensible for a fourth-year research project by
adding:

- temporal harvest-session grouping;
- session-balanced positive sample weights;
- four prespecified feature ablations;
- full-prevalence validation and test evaluation;
- event-level recall and lead-time analysis;
- false-alert episode counting;
- grouped leave-one-positive-hive-out robustness analysis;
- one untouched test evaluation after selecting the candidate.

## Important scientific position

The four models are appropriate for a benchmark comparison. The selected
candidate is the best-performing candidate **within this dataset**, not proof
of a universally optimal harvest model.

The current target means:

> a manually reviewed probable harvest event occurs within the next 72 hours.

It does not independently verify honey maturity or the globally optimal
extraction time.

## Feature-set comparison

Every enabled model is evaluated with:

- `core`: 15 prespecified features;
- `weight_only`: all weight-derived features;
- `no_humidity`: all features except generated humidity features;
- `full`: all 63 features.

This creates a model-and-ablation comparison instead of selecting a model from
one potentially overfitted feature set.

## Installation

Extract the ZIP at the repository root.

Append:

`backend/config/harvesting_research_models_section.yaml`

to:

`backend/config/harvesting.yaml`

Then delete the snippet file.

Install optional model libraries:

```powershell
pip install xgboost lightgbm
```

## Run

From `backend`:

```powershell
ruff check . --fix
ruff check .
pytest -v

python scripts/run_research_harvest_model_comparison.py
```

## Outputs

Reports:

```text
artifacts/reports/harvesting/reviewed/research_models/
├── model_feature_set_comparison.csv
├── selected_model_metrics.json
├── selected_validation_predictions.parquet
├── selected_test_predictions.parquet
├── selected_threshold_sweep.csv
├── selected_validation_event_detection.csv
├── selected_test_event_detection.csv
├── selected_grouped_hive_robustness.csv
├── selected_feature_importance.csv
├── reviewed_events_with_sessions.csv
└── harvest_session_summary.csv
```

Model artifact:

```text
artifacts/models/harvesting/research_v2/
├── selected_model.joblib
├── selected_feature_columns.json
└── model_metadata.json
```

## Selection rule

The script:

1. requires complete validation event recall when possible;
2. ranks remaining candidates by validation PR-AUC;
3. prefers fewer false-alert episodes;
4. prefers fewer features;
5. prefers the simpler model in a tie.

Accuracy is not used as the primary metric.

## Stop point

Do not show `raw_probability × 100` as HUI yet.

The next milestone uses grouped out-of-fold predictions from the official
training split to create provisional sigmoid calibration. Only after that
stage should the system calculate:

- HUI;
- readiness classes;
- HRSI;
- HRRoC;
- candidate harvest windows;
- live API outputs.
