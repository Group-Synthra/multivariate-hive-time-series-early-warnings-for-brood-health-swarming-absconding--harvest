# Harvest Probability Calibration Package

This package implements the next final-viva stage after the completed
72-hour classification comparison.

## What it does

1. Loads the selected XGBoost / no-humidity classifier definition.
2. Builds training-only grouped out-of-fold raw probabilities.
3. Keeps every hive inside one calibration fold.
4. Fits identity, Platt and isotonic calibration candidates.
5. Selects the method only from validation Brier score, log loss and ECE.
6. Evaluates the frozen choice once on the held-out test split.
7. Saves the calibrator, calibrated prediction files, reliability bins,
   comparison metrics and a research gate.

It does not yet create HUI values. The exact HUI mapping must be agreed
after the calibrated probability distribution is inspected.

## Installation

Extract the ZIP at the project root so that its `backend` folder merges with
the existing `backend` folder.

From `backend`:

```powershell
python scripts/merge_harvesting_probability_calibration_config.py
```

Validate the merged section:

```powershell
python -c "import yaml; c=yaml.safe_load(open('config/harvesting.yaml',encoding='utf-8')); print(c['probability_calibration'])"
```

The temporary section file may then be removed:

```powershell
Remove-Item `
  .\config\harvesting_probability_calibration_section.yaml `
  -ErrorAction SilentlyContinue
```

## Quality checks

```powershell
ruff check . --fix
ruff check .

pytest `
  tests/modules/harvesting/test_probability_calibration.py `
  -v

pytest -v
```

## Run calibration

```powershell
python scripts/run_harvest_probability_calibration.py
```

Main outputs:

```text
artifacts/reports/harvesting/reviewed/probability_calibration/
  calibration_method_comparison.csv
  calibration_reliability_bins.csv
  grouped_oof_fold_audit.csv
  training_oof_calibrated_predictions.parquet
  selected_validation_calibrated_predictions.parquet
  selected_test_calibrated_predictions.parquet
  probability_calibration_summary.json
  probability_calibration_gate.json

artifacts/models/harvesting/probability_calibration/
  selected_probability_calibrator.joblib
  probability_calibrator_metadata.json
```

## Print the final comparison

```powershell
@'
import pandas as pd

path = (
    "artifacts/reports/harvesting/reviewed/"
    "probability_calibration/"
    "calibration_method_comparison.csv"
)
data = pd.read_csv(path)

columns = [
    "method",
    "split",
    "brier_score",
    "log_loss",
    "expected_calibration_error",
    "calibration_intercept",
    "calibration_slope",
    "pr_auc",
    "roc_auc",
    "mean_probability",
    "maximum_probability",
]

print(
    data.loc[data["status"].eq("ok"), columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)
'@ | python -
```

## Print the research gate

```powershell
python -c "import json; p='artifacts/reports/harvesting/reviewed/probability_calibration/probability_calibration_gate.json'; print(json.dumps(json.load(open(p,encoding='utf-8')),indent=2))"
```

## Inspect candidate HUI ranges without adopting a formula

This only prints calibrated probability percentages. It does not permanently
define HUI.

```powershell
@'
import pandas as pd

root = (
    "artifacts/reports/harvesting/reviewed/"
    "probability_calibration"
)

for split in ("validation", "test"):
    path = (
        f"{root}/selected_{split}_"
        "calibrated_predictions.parquet"
    )
    data = pd.read_parquet(path)
    percent = data["calibrated_probability"] * 100.0

    print(f"\n{split.upper()}")
    print(
        percent.describe(
            percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]
        ).round(4)
    )
'@ | python -
```

## Interpretation

A passed research gate supports proceeding to a provisional academic HUI
definition. It never enables operational deployment because validation has
two reviewed events and test has one reviewed event.

A failed gate means the code completed but the available evidence did not
show that calibration improved probability quality sufficiently. Do not
change metrics or thresholds merely to make the gate pass.
