# Harvesting Model Dashboard Integration

The research comparison is not displayed automatically. The existing Model
Training tab is a placeholder until the exported model results and React
component are connected.

## Files added

```text
backend/scripts/export_harvest_model_results_for_frontend.py

frontend/src/services/harvestingModelService.js
frontend/src/features/harvesting/model/
├── HarvestingModelTrainingTab.jsx
├── HarvestingModelTrainingTab.css
└── index.js

frontend/src/features/harvesting/HarvestingPage.jsx
frontend/public/data/harvesting-models/.gitkeep
```

## Export the completed model results

From `backend`:

```powershell
python scripts/export_harvest_model_results_for_frontend.py
```

Expected output file:

```text
frontend/public/data/harvesting-models/dashboard.json
```

## Start frontend

```powershell
cd ..\frontend
npm run dev
```

Open:

```text
4. Harvesting
→ Model Training
```

## Displayed sections

- selected model and feature set;
- validation PR-AUC;
- validation and test event detection;
- false-alert episodes;
- all 16 model/feature-set candidates;
- grouped-hive robustness;
- feature importance;
- research limitations;
- next-stage workflow.

## Research restriction

The dashboard intentionally labels model outputs as uncalibrated scores.
Do not display HUI, readiness classes, HRSI, HRRoC or a harvest recommendation
until the calibration and live-inference milestone is complete.
