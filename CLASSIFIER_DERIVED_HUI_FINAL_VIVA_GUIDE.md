# Classifier-Derived HUI Final Viva Integration

This package adds the final viva research dashboard for the harvesting module.

It uses the completed results:

- Current HUI derived from the selected 72-hour XGBoost classifier.
- Platt score transformation used only as a research-stage input to HUI.
- Future HUI regression at 24, 48 and 72 hours.
- Future-HUI research gate passed at all three horizons.
- Operational deployment remains disabled.

## Final evaluation values

| Horizon | Selected model | Feature set | Validation MAE | Test MAE | Test RMSE | Test R² | Test within ±5 | Test class agreement |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 24h | LightGBM | No Humidity + HUI | 3.071 | 3.754 | 5.168 | 0.773 | 72.8% | 90.0% |
| 48h | XGBoost | No Humidity + HUI | 3.851 | 4.506 | 6.457 | 0.647 | 67.1% | 87.5% |
| 72h | LightGBM | No Humidity + HUI | 4.654 | 5.280 | 7.666 | 0.508 | 62.6% | 86.2% |

Validation MAE improvement over persistence:

- 24h: 33.8%
- 48h: 47.4%
- 72h: 48.4%

These values support the final viva research dashboard. They do not establish independently verified honey maturity or operational deployment.

## Package contents

```text
backend/scripts/export_classifier_derived_hui_viva_dashboard.py
backend/tests/modules/harvesting/test_classifier_derived_hui_viva_export.py
frontend/src/services/classifierDerivedHuiService.js
frontend/src/features/harvesting/live/ClassifierDerivedHuiPredictionTab.jsx
frontend/src/features/harvesting/live/ClassifierDerivedHuiPredictionTab.css
frontend/src/features/harvesting/model/ClassifierDerivedHuiEvaluationPanel.jsx
frontend/src/features/harvesting/model/ClassifierDerivedHuiEvaluationPanel.css
frontend/src/features/harvesting/model/HarvestingFinalResearchPanel.jsx
frontend/src/features/harvesting/model/HarvestingFinalResearchPanel.css
apply_classifier_derived_hui_frontend_integration.py
```

## 1. Extract from the project root

```powershell
cd "C:\Users\user\Desktop\Research\multivariate-hive-time-series-early-warnings-for-brood-health-swarming-absconding--harvest"

$package = Get-ChildItem `
  "$HOME\Downloads" `
  -Filter "classifier_derived_hui_viva_frontend_package*.zip" |
Sort-Object LastWriteTime -Descending |
Select-Object -First 1

$package.FullName

Expand-Archive `
  -Path $package.FullName `
  -DestinationPath . `
  -Force
```

## 2. Run backend quality checks

```powershell
cd .\backend\

ruff check . --fix
ruff check .

pytest `
  tests/modules/harvesting/test_classifier_derived_hui_viva_export.py `
  -v

pytest -v
```

The new file contains five tests. The total test count should increase from 78 to 83.

## 3. Export the final viva dashboard JSON

```powershell
python `
  scripts/export_classifier_derived_hui_viva_dashboard.py
```

Expected structure:

```json
{
  "status": "classifier_derived_hui_viva_dashboard_exported",
  "hive_count": 48,
  "series_rows": 8064,
  "future_hui_gate_passed": true,
  "operational_deployment_allowed": false
}
```

Confirm the file:

```powershell
Test-Path `
  ..\frontend\public\data\harvesting-research\classifier-derived-hui-viva-dashboard.json
```

Expected:

```text
True
```

## 4. Apply frontend integration

Move to the project root:

```powershell
cd ..

python `
  .\apply_classifier_derived_hui_frontend_integration.py
```

The script:

- Replaces the experimental Provisional HUI tab with `ClassifierDerivedHuiPredictionTab`.
- Changes the visible tab label to `HUI Decision Support`.
- Adds the final calibration and future-HUI evaluation panel to the Model Training tab.
- Replaces the old final-research panel that said HUI was unsupported.
- Creates backups using `.before_classifier_hui`.

## 5. Verify the imports and render blocks

```powershell
Select-String `
  -Path .\frontend\src\features\harvesting\HarvestingPage.jsx `
  -Pattern `
    "ClassifierDerivedHuiPredictionTab", `
    "ProvisionalHuiPredictionTab", `
    "ExploratoryWeightForecastTab"
```

Only `ClassifierDerivedHuiPredictionTab` should appear.

```powershell
Select-String `
  -Path .\frontend\src\features\harvesting\model\HarvestingModelTrainingTab.jsx `
  -Pattern "ClassifierDerivedHuiEvaluationPanel"
```

It should appear in the import and render call.

## 6. Build the frontend

```powershell
cd .\frontend\

npm run build
npm run dev
```

Open:

```text
http://localhost:5173/
```

Use:

```text
Harvesting → Model Training
Harvesting → HUI Decision Support
```

## Model Training viva content

The existing 16-candidate model comparison table remains unchanged.

The new evaluation section adds:

- Raw versus Platt Brier score, log loss, ECE and calibration slope.
- The reason the calibration gate remained research-stage.
- 24h, 48h and 72h regression MAE, R², tolerance and class agreement.
- A clear statement that the future-HUI gate passed for the viva prototype.

## HUI Decision Support outputs

The final tab displays:

1. Current HUI.
2. 24h, 48h and 72h predicted HUI.
3. Harvest-readiness classes.
4. Harvest Readiness Stability Index.
5. HUI rate of change.
6. Recommended inspection/harvest window.
7. Research prediction confidence.
8. Current sensor and environmental status available in the dataset.
9. Final recommendation.
10. Explanation of contributing factors.

The exported screen uses held-out historical test records. It is a valid model demonstration for the viva, but it is not the live IoT ingestion API.

## Live IoT distinction

For actual live inference, the backend must receive the latest continuous sensor history for the hive, build the same 53 model features, load:

```text
selected_model.joblib
selected_probability_calibrator.joblib
selected_classifier_derived_hui_regressor_24h.joblib
selected_classifier_derived_hui_regressor_48h.joblib
selected_classifier_derived_hui_regressor_72h.joblib
```

and return the same output contract used by this dashboard.

That API wiring depends on the project’s current backend framework, database tables and IoT endpoint. Do not claim that the static held-out dashboard is already receiving live sensor messages.

## Viva wording

> Four classifiers were compared across four feature sets using PR-AUC as the primary metric for the highly imbalanced harvest-event target. XGBoost without humidity was selected. Platt scaling improved Brier score but did not pass the complete calibration gate because validation ECE increased slightly, so it was used only to construct a classifier-derived provisional HUI. LightGBM and XGBoost regressors then forecast HUI at 24, 48 and 72 hours. All three horizons improved validation MAE over persistence and passed the predefined future-HUI research gate. The outputs support a final decision-support research prototype, while independent biological and operational validation remains future work.
