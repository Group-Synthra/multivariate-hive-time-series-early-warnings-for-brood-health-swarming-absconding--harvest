# Provisional HUI Regression — Complete Implementation Guide

## Research definition

This package predicts a continuous **Provisional Harvest Utilization Index
(Provisional HUI)** from 0 to 100.

It does not train against verified honey maturity because no verified HUI
ground-truth column exists. Instead:

1. A transparent current Provisional HUI is engineered from current and past
   telemetry.
2. Future Provisional HUI targets are obtained from the engineered HUI at exact
   future timestamps.
3. Regression models predict future Provisional HUI at 24, 48 and 72 hours.
4. Fixed display thresholds convert predicted values into readiness classes.

The output remains a research prototype and is not a calibrated harvest
probability.

## HUI components

| Component | Weight | Direction |
|---|---:|---|
| Weight relative to recent 168h maximum | 30% | Higher is better |
| Positive 72h weight accumulation | 25% | Higher is better |
| Absolute 72h weight trend | 25% | Lower indicates plateau |
| Environmental variability | 10% | Lower is better |
| 24h temperature range | 10% | Lower is better |

CO2 flatline flags reduce the data-quality factor.

Normalization bounds use the official training split only.

## Readiness classes

| Provisional HUI | Display class |
|---:|---|
| 0 to below 40 | Not Ready |
| 40 to below 60 | Approaching Harvest |
| 60 to below 80 | Ready — Inspection Recommended |
| 80 to 100 | High-Priority Harvest Review |

## Model comparison

Each future horizon compares:

- Persistence HUI baseline
- Ridge Regression
- Random Forest Regression
- XGBoost Regression
- LightGBM Regression

Feature sets:

- Weight Only
- No Humidity

Primary model-selection metric:

- Validation MAE in HUI points

Secondary metrics:

- RMSE
- Median absolute error
- Bias
- R²
- Fraction within ±5 HUI points
- Fraction within ±10 HUI points

## Run order

From `backend`:

```powershell
python scripts/merge_harvesting_provisional_hui_regression_config.py

ruff check . --fix
ruff check .

pytest tests/modules/harvesting/test_provisional_hui_regression.py -v
pytest -v

python scripts/build_provisional_hui_dataset.py

python scripts/run_provisional_hui_regression.py

python scripts/export_provisional_hui_dashboard.py
```

## Generated backend outputs

```text
data/processed/
└── provisional_hui_dataset.parquet

artifacts/reports/harvesting/reviewed/provisional_hui_regression/
├── provisional_hui_definition.json
├── provisional_hui_distribution.csv
├── provisional_hui_regression_comparison.csv
├── provisional_hui_regression_summary.json
├── provisional_hui_regression_gate.json
├── selected_validation_predictions_24h.parquet
├── selected_validation_predictions_48h.parquet
├── selected_validation_predictions_72h.parquet
├── selected_test_predictions_24h.parquet
├── selected_test_predictions_48h.parquet
└── selected_test_predictions_72h.parquet

artifacts/models/harvesting/provisional_hui_regression/
├── selected_provisional_hui_regressor_24h.joblib
├── selected_provisional_hui_regressor_24h.json
├── selected_provisional_hui_regressor_48h.joblib
├── selected_provisional_hui_regressor_48h.json
├── selected_provisional_hui_regressor_72h.joblib
└── selected_provisional_hui_regressor_72h.json
```

## Generated frontend data

```text
frontend/public/data/harvesting-research/
└── provisional-hui-dashboard.json
```

## Frontend integration

Import the new tab in `HarvestingPage.jsx`:

```jsx
import ProvisionalHuiPredictionTab from "./live/ProvisionalHuiPredictionTab";
```

Replace the current exploratory forecast tab rendering:

```jsx
{activeModuleTab === "live-early-warning" && (
  <ProvisionalHuiPredictionTab />
)}
```

Keep the internal tab ID `live-early-warning` if it is already used for
routing. Change the visible label to:

```text
Provisional HUI Prediction
```

## Required wording

Use:

- Current Provisional HUI
- Predicted 24h Provisional HUI
- Predicted 48h Provisional HUI
- Predicted 72h Provisional HUI
- Ready — Inspection Recommended
- High-Priority Harvest Review
- Research-only index

Do not use:

- Calibrated HUI
- Harvest probability
- Guaranteed readiness
- Automatic harvest recommendation
- Verified honey maturity

## Research gate

The package records a regression gate based on improvement over persistence.

Passing the regression gate permits display in the research dashboard only.
It does not permit operational deployment because the target is an engineered
index rather than verified honey maturity.
