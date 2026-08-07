# Classifier-Derived HUI Final Viva Polish

This patch freezes the final viva wording without changing any trained model,
calibration artifact, HUI anchor, regression model, or evaluation value.

## Changes

- Renames **Prediction confidence** to **Evidence confidence**.
- Prevents a **High** confidence label while the probability-calibration gate
  remains limited.
- Caps the limited-gate prototype evidence score at 74.9 (Moderate).
- Keeps High confidence available only when the calibration gate passes.
- Clarifies that the recommended window is an inspection/potential-harvest
  window requiring beekeeper confirmation.
- Removes misleading "live-history" wording from historical demonstrations.
- Adds top-level dashboard compatibility fields:
  `research_gate` and `research_status`.
- Replaces deprecated `Timestamp.utcnow()` with `Timestamp.now(tz="UTC")`.
- Formats LightGBM and XGBoost names consistently in the HUI projection cards.

## Apply

From the project root:

```powershell
$patch = "$HOME\Downloads\classifier_derived_hui_final_viva_polish.zip"

Expand-Archive `
  -Path $patch `
  -DestinationPath . `
  -Force
```

## Validate backend

```powershell
cd .\backend\

ruff check . --fix
ruff check .

pytest `
  tests/modules/harvesting/test_classifier_derived_hui_viva_export.py `
  -v

pytest -v
```

Expected targeted result: `6 passed`.

The full suite should increase from 83 to approximately `84 passed`.

## Re-export the dashboard

```powershell
python `
  scripts/export_classifier_derived_hui_viva_dashboard.py
```

Verify the compatibility fields:

```powershell
@'
import json

path = (
    "../frontend/public/data/harvesting-research/"
    "classifier-derived-hui-viva-dashboard.json"
)

with open(path, encoding="utf-8") as file:
    data = json.load(file)

print("STATUS:", data["status"])
print("FUTURE HUI GATE:", data["research_gate"]["gate_passed"])
print(
    "OPERATIONAL DEPLOYMENT:",
    data["research_status"]["operational_deployment_allowed"],
)
print(
    "CALIBRATION OPERATIONALLY VALIDATED:",
    data["research_status"][
        "probability_calibration_operationally_validated"
    ],
)
'@ | python -
```

Expected:

```text
FUTURE HUI GATE: True
OPERATIONAL DEPLOYMENT: False
CALIBRATION OPERATIONALLY VALIDATED: False
```

## Build frontend

```powershell
cd ..
cd .\frontend\

npm run build
npm run dev
```

## Final screen checks

The HUI Decision Support screen should show:

- **Evidence confidence**
- A maximum label of **Moderate** while calibration is limited
- **Recommended inspection / potential harvest window**
- Historical-demo wording rather than a claim of a live stream

No model retraining is required.
