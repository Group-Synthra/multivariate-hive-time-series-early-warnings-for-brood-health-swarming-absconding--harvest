# Absconding Module — Report-Aligned UI and Supabase IoT Update

This patch updates the Absconding module in the BeeHive monorepo. It preserves the new shared-data architecture while restoring the three report-facing interfaces:

1. **Exploratory Analysis** — hive selection, risk percentage, ARM, environmental stress, risk timeline, sensor behaviour and explainable factors.
2. **Model Training** — compatible report model families, validation comparison, active model metrics, feature importance, confusion matrix and temporal-model rationale.
3. **Live Prediction (IoT)** — a real Supabase PostgreSQL read, saved-model inference, ARM, freshness, sensor cards, contributing factors, notification and beekeeper actions.

## Security first

Do not commit `backend/.env`. The source and patch contain only `backend/.env.example` with placeholders. If a real database password was pasted into a chat, issue tracker, screenshot or commit, rotate it in Supabase before using the system.

## 1. Apply the patch

Create or switch to your feature branch from the repository root:

```powershell
git switch feature/absconding-v2
```

Copy everything inside `BeeHive_absconding_iot_ui_patch` into the root of the cloned BeeHive repository and allow file replacement.

## 2. Create the real environment file

From the repository root:

```powershell
Copy-Item .\backend\.env.example .\backend\.env
notepad .\backend\.env
```

Set the real values in `backend/.env`. The important groups are:

```dotenv
DATABASE_URL=postgresql://USERNAME:PASSWORD@HOST:5432/postgres
DATABASE_SSLMODE=require

IOT_MONITOR_ENABLED=true
IOT_INTERVAL_MINUTES=10
IOT_HISTORY_HOURS=192
IOT_HIVE_ID=

IOT_SCHEMA=public
IOT_SENSOR_TABLE=beehive_readings
IOT_HIVE_COLUMN=device_id
IOT_TIMESTAMP_COLUMN=recorded_at
IOT_TEMPERATURE_COLUMN=internal_temp
IOT_HUMIDITY_COLUMN=internal_humidity
IOT_CO2_COLUMN=internal_co2
IOT_WEIGHT_COLUMN=total_weight
IOT_EXTERNAL_TEMPERATURE_COLUMN=external_temp
IOT_EXTERNAL_HUMIDITY_COLUMN=external_humidity
IOT_BATTERY_VOLTAGE_COLUMN=battery_voltage
IOT_TIMESTAMPS_ARE_UTC=true
IOT_FEATURE_TIMEZONE=Asia/Colombo
```

`IOT_READING_AT_COLUMN` remains available as a fallback only when `IOT_TIMESTAMP_COLUMN` is blank. With the supplied table mapping, `recorded_at` is used.

Leave `IOT_HIVE_ID` blank to select the device with the newest database reading automatically. Set it to one specific `device_id` to lock the live dashboard to that hive.

## 3. Refresh backend dependencies

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -e ".[dev]"
python -m pip check
```

The update adds `psycopg[binary]` for direct PostgreSQL access.

## 4. Regenerate the shared data when needed

```powershell
python scripts/run_common_pipeline.py --input "data/raw/Common_Beehive_Complete_Training_Dataset_311044.xlsx"
```

Required files:

```text
data/processed/common_clean.parquet
data/manifests/common_split_manifest.parquet
```

## 5. Retrain Absconding

Retraining is required because the new module uses a leakage-safe next-24-hour target and an updated feature contract.

```powershell
python scripts/run_absconding_pipeline.py
```

Compatible model families from the previous report are included and retrained: environmental-stress baseline, Gaussian Naive Bayes, Logistic Regression, Ridge Classifier, Decision Tree, Random Forest and Extra Trees. The new pipeline also keeps a prior baseline and Isolation Forest anomaly baseline.

The old serialized LSTM is **not silently reused** because its input names, scaler/sequence contract, target definition and library version do not match the new shared pipeline. Reusing it would produce scientifically invalid output. LSTM should be retrained as a separate controlled experiment before its new-version metrics are reported.

## 6. Validate the backend

```powershell
python -m pytest
python -m ruff check src tests scripts app.py
python -m pip check
```

## 7. Start the backend and IoT poller

```powershell
python app.py
```

When `DATABASE_URL` exists and `IOT_MONITOR_ENABLED=true`, the backend monitor starts in the request-serving process and performs a fresh database prediction every 10 minutes. It caches only the latest prediction JSON under the ignored `backend/artifacts/predictions/absconding/` directory.

Useful endpoints:

```text
GET  /api/absconding/summary
GET  /api/absconding/metrics
GET  /api/absconding/hives
POST /api/absconding/predict
GET  /api/absconding/iot/live
GET  /api/absconding/iot/live?force=true
GET  /api/absconding/iot/monitor/status
POST /api/absconding/iot/monitor/run-now
POST /api/absconding/iot/monitor/start
POST /api/absconding/iot/monitor/stop
```

Quick PowerShell checks:

```powershell
Invoke-RestMethod "http://127.0.0.1:5000/api/absconding/iot/monitor/status"
Invoke-RestMethod "http://127.0.0.1:5000/api/absconding/iot/live?force=true"
```

The model requires 168 hourly observations. Ten-minute database readings are mapped and aggregated to hourly means. Until enough history exists, the API returns `status: collecting_history` and does not invent a probability.

## 8. Start the frontend

In another terminal:

```powershell
cd frontend
npm.cmd install
npm.cmd run build
npm.cmd run dev
```

Open `http://localhost:5173`, select **Absconding**, then open **Live Prediction (IoT)**.

## Where to change each part later

| Requirement | File |
|---|---|
| Database URL and IoT column mapping | `backend/.env` |
| Safe environment template | `backend/.env.example` |
| PostgreSQL query and automatic hive selection | `backend/src/multivari/modules/absconding/iot.py` |
| 10-minute polling and cache | `backend/src/multivari/modules/absconding/iot_monitor.py` |
| Live preprocessing, inference, ARM, freshness, factors and actions | `backend/src/multivari/modules/absconding/service.py` |
| Live and monitor endpoints | `backend/src/multivari/modules/absconding/routes.py` |
| Monitor creation | `backend/src/multivari/api/app_factory.py` and `backend/app.py` |
| Historical feature engineering | `backend/src/multivari/modules/absconding/features.py` |
| Model candidates | `backend/src/multivari/modules/absconding/modeling.py` and `backend/config/absconding.yaml` |
| Exploratory and training interfaces | `frontend/src/features/absconding/AbscondingPage.jsx` |
| Live IoT interface | `frontend/src/features/absconding/AbscondingLiveDashboard.jsx` |
| Frontend polling/API calls | `frontend/src/hooks/useAbscondingData.js` and `frontend/src/services/abscondingApi.js` |
| Dashboard styling | `frontend/src/styles/index.css` |

## Operational limitation

The historical workbook contains very few distinct confirmed Absconding episodes. The pipeline therefore remains an exploratory early-warning prototype. The IoT connection verifies data retrieval and live inference, but local biological validation requires more correctly labelled Sri Lankan Absconding events.
