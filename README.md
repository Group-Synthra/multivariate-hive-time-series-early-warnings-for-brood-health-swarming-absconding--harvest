# Harvesting EDA Frontend Integration

The uploaded frontend already contains the EDA component, service, exported JSON files and figures.
The missing step was rendering the EDA component from the harvesting module page.

## Install

Extract this ZIP at the repository root with overwrite enabled.

It replaces only:

`frontend/src/features/harvesting/HarvestingPage.jsx`

## Run

```powershell
cd frontend
npm run dev
```

Open **4. Harvesting**, then **Exploratory Analysis**.

The other two module tabs remain clearly marked as future milestones until model training and live inference are implemented.
Move the existing Vite application into `frontend/`. Never commit `.env`, `node_modules`, raw datasets, or trained model binaries.

## Absconding module

Absconding uses a separate labelled historical dataset so the common dataset and workflows
for Brood Health, Swarming and Harvesting remain unchanged.

Copy the Absconding CSV to:

```text
backend/data/raw/absconding/hive_data_with_features.csv
```

Then run:

```bash
cd backend
python scripts/run_absconding_data_pipeline.py
python scripts/run_absconding_pipeline.py
```

The generated module-specific files are:

- `data/processed/absconding_clean.parquet`
- `data/manifests/absconding_split_manifest.parquet`
- `artifacts/reports/absconding/absconding_data_profile.json`

The Absconding workspace provides **Exploratory Analysis**, **Model Training**, and
**Live Prediction (IoT)** views. Live inference reads Supabase PostgreSQL history, maps the
IoT columns, aggregates ten-minute readings to hourly features, loads the saved model,
calculates ARM, and returns risk, explanations, freshness and recommended actions.

Optional LSTM training uses 72-observation sequences with stride 3:

```bash
python -m pip install -e ".[dev,lstm]"
python scripts/run_absconding_lstm.py --epochs 30 --sequence-length 72 --stride 3
python scripts/run_absconding_pipeline.py
```

See [`ABSCONDING_SEPARATE_DATASET_UPDATE_GUIDE.md`](ABSCONDING_SEPARATE_DATASET_UPDATE_GUIDE.md)
and [`ABSCONDING_IOT_UI_UPDATE_GUIDE.md`](ABSCONDING_IOT_UI_UPDATE_GUIDE.md).
