# Absconding Separate-Dataset Update Guide

## Purpose

This update changes only the Absconding model-development data source. The other modules
continue using:

```text
backend/data/processed/common_clean.parquet
backend/data/manifests/common_split_manifest.parquet
```

Absconding now uses:

```text
backend/data/processed/absconding_clean.parquet
backend/data/manifests/absconding_split_manifest.parquet
```

The trained model still integrates normally with the same Flask API, React interface and
Supabase IoT pipeline.

## Supplied dataset result

The uploaded `hive_data_with_features.csv` contains 277,376 hourly records from 48 hives.
It contains 718 rows where the event is active. The module converts active intervals into 89
onset markers and then merges nearby onsets into 58 event episodes for event-level evaluation.

The source 72-hour target is retained only for validation during data preparation and is not
included in the model feature table. The production target is derived from event onsets:

```text
absconding_within_24h = 1 when an event onset occurs in the next 24 hourly readings
```

## Files in this patch

### New

```text
backend/scripts/run_absconding_data_pipeline.py
backend/src/multivari/modules/absconding/data.py
backend/tests/test_absconding_data.py
ABSCONDING_SEPARATE_DATASET_UPDATE_GUIDE.md
```

### Updated

```text
backend/config/absconding.yaml
backend/scripts/run_absconding_pipeline.py
backend/src/multivari/modules/absconding/__init__.py
backend/src/multivari/modules/absconding/config.py
backend/src/multivari/modules/absconding/features.py
backend/src/multivari/modules/absconding/pipeline.py
backend/src/multivari/modules/absconding/service.py
backend/src/multivari/modules/absconding/lstm.py
backend/src/multivari/modules/absconding/README.md
backend/tests/test_absconding_module.py
backend/tests/test_absconding_iot.py
frontend/src/features/absconding/AbscondingPage.jsx
README.md
```

## 1. Copy the dataset

Create the folder:

```powershell
New-Item -ItemType Directory -Force ".\backend\data\raw\absconding" | Out-Null
```

Copy the CSV:

```powershell
Copy-Item "$HOME\Downloads\hive_data_with_features.csv" ".\backend\data\raw\absconding\hive_data_with_features.csv"
```

## 2. Install or refresh backend dependencies

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pip check
```

For LSTM:

```powershell
python -m pip install -e ".[dev,lstm]"
```

TensorFlow is optional. The classical pipeline and live tabular inference work without it.

## 3. Prepare only the Absconding data

```powershell
python scripts/run_absconding_data_pipeline.py
```

Expected profile values for the uploaded CSV:

```text
Rows: 277,376
Hives: 48
Event onset markers: 89
Merged event episodes: 58
```

Verify:

```powershell
Test-Path ".\data\processed\absconding_clean.parquet"
Test-Path ".\data\manifests\absconding_split_manifest.parquet"
Test-Path ".\artifacts\reports\absconding\absconding_data_profile.json"
```

## 4. Train classical models

```powershell
python scripts/run_absconding_pipeline.py
```

A quicker code check is:

```powershell
python scripts/run_absconding_pipeline.py --models rule_based_stress logistic_balanced extra_trees
```

## 5. Train LSTM

First verify TensorFlow:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

Then test with three epochs:

```powershell
python scripts/run_absconding_lstm.py --epochs 3 --sequence-length 72 --stride 3
```

Full run:

```powershell
python scripts/run_absconding_lstm.py --epochs 30 --sequence-length 72 --stride 3
```

Merge the LSTM comparison into the dashboard:

```powershell
python scripts/run_absconding_pipeline.py
```

## 6. Validate

```powershell
python -m ruff check src tests scripts app.py --fix
python -m ruff format src tests scripts app.py
python -m ruff check src tests scripts app.py
python -m pytest
```

## 7. Start application

Backend:

```powershell
python app.py
```

Frontend in a second terminal:

```powershell
cd ..\frontend
npm.cmd install
npm.cmd run dev
```

Open:

```text
http://localhost:5173
```

## Integration design and validation

No code in the common data pipeline is redirected to the new CSV. Brood Health, Swarming and
Harvesting remain attached to the common dataset. Only `AbscondingPaths.clean_data` and
`AbscondingPaths.split_manifest` point to the module-specific files. The API route names and
frontend response structure remain unchanged, and the update was validated against the existing API and frontend response contract.
