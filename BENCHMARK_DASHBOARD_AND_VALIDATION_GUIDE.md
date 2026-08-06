# Harvesting Benchmark Dashboard and Prospective Validation

## Current research position

The classifier alert-policy gate failed. The robust forecasting gate also
failed because only the 24-hour horizon improved over persistence, while the
48- and 72-hour horizons selected persistence.

The 24-hour result may be shown only as an exploratory hive-weight forecast.

Do not create HUI, HRSI, HRRoC, readiness classes or harvest recommendations.

## Files in this package

```text
backend/scripts/
├── finalize_harvesting_benchmark_state.py
├── create_prospective_harvest_validation_template.py
└── export_harvesting_benchmark_dashboard.py

backend/tests/modules/harvesting/
└── test_benchmark_dashboard_export.py

frontend/src/services/
└── harvestingBenchmarkService.js

frontend/src/features/harvesting/model/
├── HarvestingFinalResearchPanel.jsx
└── HarvestingFinalResearchPanel.css

frontend/src/features/harvesting/live/
├── ExploratoryWeightForecastTab.jsx
└── ExploratoryWeightForecastTab.css
```

## Frontend integration

### Model Training tab

Import:

```jsx
import HarvestingFinalResearchPanel from "./HarvestingFinalResearchPanel";
```

The tab already loads `dashboard`. Render:

```jsx
<HarvestingFinalResearchPanel dashboard={dashboard} />
```

Place it after the model dashboard heading.

### Harvesting page

Import:

```jsx
import ExploratoryWeightForecastTab from "./live/ExploratoryWeightForecastTab";
```

Replace the Live Early Warning placeholder with:

```jsx
{activeModuleTab === "live-early-warning" && (
  <ExploratoryWeightForecastTab />
)}
```

The existing tab name may remain for project navigation, but the page itself
must clearly state that it is a research-only weight forecast.

## Export

From `backend`:

```powershell
python scripts/finalize_harvesting_benchmark_state.py
python scripts/create_prospective_harvest_validation_template.py
python scripts/export_harvesting_benchmark_dashboard.py
```

## Required dashboard wording

Approved:

- Benchmark-only classifier
- Alert-policy gate failed
- Robust forecasting gate failed
- Exploratory 24-hour hive-weight forecast
- Prospective validation required

Blocked:

- Harvest probability
- HUI
- Ready to harvest
- Recommended harvest time
- Live harvesting recommendation
