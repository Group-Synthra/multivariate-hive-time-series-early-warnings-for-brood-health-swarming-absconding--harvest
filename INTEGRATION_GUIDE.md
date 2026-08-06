# Harvesting EDA Dashboard Integration

This package is additive. It does not overwrite the existing `App.jsx`.

## Added files

```text
backend/scripts/export_harvest_eda_for_frontend.py

frontend/src/services/harvestingEdaService.js
frontend/src/features/harvesting/eda/HarvestingEdaTab.jsx
frontend/src/features/harvesting/eda/HarvestingEdaTab.css
frontend/src/features/harvesting/eda/index.js
frontend/public/data/harvesting/.gitkeep
```

## 1. Extract

Extract this ZIP at the repository root, not inside `frontend`.

## 2. Generate the reviewed EDA outputs first

From `backend`:

```powershell
python scripts/run_reviewed_harvest_feature_eda.py
python scripts/build_grouped_hive_validation.py
```

## 3. Export EDA data to the frontend

From `backend`:

```powershell
python scripts/export_harvest_eda_for_frontend.py
```

This creates:

```text
frontend/public/data/harvesting/
├── summary.json
├── top-features.json
├── feature-comparison.json
├── sample-coverage.json
├── manifest.json
└── figures/
```

Run the export command again whenever backend EDA results change.

## 4. Connect the component to the existing EDA tab

Use this import in the current harvesting page or tab container:

```jsx
import HarvestingEdaTab from "./features/harvesting/eda";
```

Adjust the relative path when the importing file is inside another folder.

Render it where the current EDA tab content belongs:

```jsx
{activeTab === "eda" && <HarvestingEdaTab />}
```

For switch-based tab rendering:

```jsx
case "eda":
  return <HarvestingEdaTab />;
```

Do not replace the full `App.jsx` without checking the existing routing and
layout.

## 5. Start the frontend

From `frontend`:

```powershell
npm run dev
```

No new npm package is required.

## 6. Production build

```powershell
npm run build
```

Vite automatically includes files under `public/data/harvesting`.

## Scope

The EDA tab now supports real data:

- reviewed-event summary
- target class balance
- leakage and history checks
- lead-time feature comparison
- matched-control coverage
- grouped-hive validation summary
- exported EDA figures

Real HUI, HRSI, HRRoC and harvest recommendations should be connected only
after model training, calibration and threshold selection.
