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
