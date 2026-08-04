# MULTIVARI common team pipeline

This starter reorganises the project into a clean monorepo and provides the shared data foundation for the four research modules.

## Setup

```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

Copy the workbook to `backend/data/raw/` and run:

```bash
python scripts/run_common_pipeline.py \
  --input data/raw/Common_Beehive_Complete_Training_Dataset_311044.xlsx
```

Generated local outputs:

- `data/processed/common_clean.parquet`
- `data/manifests/common_split_manifest.parquet`
- `artifacts/reports/raw_validation_report.json`
- `artifacts/reports/clean_validation_report.json`
- `artifacts/reports/common_eda/`

Run tests:

```bash
pytest
ruff check src tests scripts
```

Move the existing Vite application into `frontend/`. Never commit `.env`, `node_modules`, raw datasets, or trained model binaries.

## Absconding module

After the common pipeline has generated the cleaned parquet and split manifest, run:

```bash
cd backend
python scripts/run_absconding_pipeline.py
```

Then start the API and frontend normally. The Absconding page reads generated artifacts from `/api/absconding/summary`.

## Absconding: report-aligned UI and live Supabase IoT

The Absconding workspace provides **Exploratory Analysis**, **Model Training**, and **Live Prediction (IoT)** views. Live inference reads the configured Supabase PostgreSQL `beehive_readings` history, maps the IoT columns, aggregates ten-minute readings to hourly features, loads the saved Absconding model, calculates ARM, and returns risk, explanations, freshness and recommended actions.

See [`ABSCONDING_IOT_UI_UPDATE_GUIDE.md`](ABSCONDING_IOT_UI_UPDATE_GUIDE.md) for environment configuration, retraining, endpoints and file locations.
