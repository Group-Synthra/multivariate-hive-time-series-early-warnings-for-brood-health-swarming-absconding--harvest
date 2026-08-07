# Separate HUI Decision Support and Live IoT Prediction tabs

This patch restores the final historical/viva **HUI Decision Support** screen and adds a separate **Live IoT Prediction** screen.

## Resulting tabs

1. Exploratory Analysis
2. Model Training
3. HUI Decision Support — held-out historical research demonstration
4. Live IoT Prediction — PostgreSQL/Supabase inference and history-readiness status

The live endpoint currently returns HTTP 422 because the newest continuous segment has fewer than 192 hourly observations. That is expected. The new live tab displays the collection progress instead of replacing the historical dashboard.

## Apply

From the project root:

```powershell
$package = Get-ChildItem `
  "$HOME\Downloads" `
  -Filter "separate_hui_decision_and_live_prediction_tabs*.zip" |
Sort-Object LastWriteTime -Descending |
Select-Object -First 1

Expand-Archive `
  -Path $package.FullName `
  -DestinationPath . `
  -Force

python .\apply_separate_hui_tabs.py
```

## Verify source integration

```powershell
Select-String `
  -Path .\frontend\src\features\harvesting\HarvestingPage.jsx `
  -Pattern `
    "ClassifierDerivedHuiPredictionTab", `
    "LiveIoTHuiPredictionTab", `
    "live-early-warning", `
    "live-iot-prediction"

Select-String `
  -Path .\frontend\src\features\shared\ModuleTabs.jsx `
  -Pattern `
    "HUI Decision Support", `
    "Live IoT Prediction"
```

## Build

```powershell
cd .\frontend
npm run build
npm run dev
```

## Expected live-tab state now

The current sensor sequence has about 85 contiguous hourly rows:

- Current-HUI requirement: 168 hours
- Full future-HUI requirement: 192 hours
- Current-HUI hours remaining: about 83
- Future-HUI hours remaining: about 107

The live tab should therefore show **Collecting contiguous history**. It must not display the older 3 August prediction as the current 6 August live result.

## Backend

No backend model or API change is needed for this tab separation. Keep the Flask server running. HTTP 422 from `/api/harvesting/live-hui` is the correct readiness response until the history requirement is met.
