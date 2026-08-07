# Live PostgreSQL → Current HUI → Future HUI Integration

This package connects `public.beehive_readings` to the completed harvesting research module.

## Frozen model flow

```text
PostgreSQL readings (approximately every 10 minutes)
    -> Sri Lanka local hourly median aggregation
    -> exact historical feature engineering
    -> selected 72-hour XGBoost classifier
    -> saved Platt score mapping
    -> classifier-derived current HUI
    -> saved future-HUI regressors
         24h LightGBM
         48h XGBoost
         72h LightGBM
    -> HRSI, rate of change, inspection window and recommendation
    -> Flask API
    -> React HUI Decision Support tab
```

No online retraining occurs. The saved viva models remain frozen.

## Confirmed database mapping

| Research field | PostgreSQL column |
|---|---|
| Hive/device | `device_id` |
| Timestamp | `recorded_at` |
| Internal temperature | `internal_temp` |
| Internal humidity | `internal_humidity` |
| Internal CO2 | `internal_co2` |
| Weight | `total_weight` |
| External temperature | `external_temp` |
| External humidity | `external_humidity` |
| Battery voltage | `battery_voltage` |

The uploaded table does not show `reading_at`; leave `IOT_READING_AT_COLUMN` blank. The repository also inspects the actual table and automatically ignores missing optional columns.

---

# 1. Create a safety checkpoint

From the project root:

```powershell
git status
git add .
git commit -m "Checkpoint before live IoT HUI integration"
```

Never commit the real PostgreSQL password.

---

# 2. Extract the package

```powershell
cd `
"C:\Users\user\Desktop\Research\multivariate-hive-time-series-early-warnings-for-brood-health-swarming-absconding--harvest"

$package = Get-ChildItem `
  "$HOME\Downloads" `
  -Filter "live_postgresql_hui_integration_sl*.zip" |
Sort-Object LastWriteTime -Descending |
Select-Object -First 1

$package.FullName

Expand-Archive `
  -Path $package.FullName `
  -DestinationPath . `
  -Force
```

Confirm:

```powershell
Test-Path `
  .\backend\src\multivari\modules\harvesting\live_hui_inference.py
```

Expected: `True`.

---

# 3. Apply project integration

```powershell
python .\apply_live_hui_integration.py
```

This safely updates:

- `backend/pyproject.toml`
- `backend/src/multivari/api/app_factory.py`
- `.gitignore`

It also verifies that the new backend and frontend files were extracted. Backups use `.before_live_hui`.

---

# 4. Configure `backend/.env`

```powershell
Copy-Item `
  .\backend\.env.live.example `
  .\backend\.env `
  -Force

code .\backend\.env
```

Use this configuration and replace only the database password/project details:

```dotenv
API_HOST=127.0.0.1
API_PORT=5000
FLASK_DEBUG=true
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

DATABASE_URL=postgresql://grafana_reader.PROJECT_REF:ENCODED_PASSWORD@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres
DATABASE_SSLMODE=require

IOT_MONITOR_ENABLED=true
IOT_INTERVAL_MINUTES=10
IOT_HISTORY_HOURS=336
IOT_HISTORY_REFERENCE=now
IOT_HIVE_ID=hive_01
IOT_STALE_AFTER_MINUTES=30
IOT_SERIES_ROWS_PER_HIVE=168
IOT_MIN_READINGS_PER_HOUR=1

IOT_SCHEMA=public
IOT_SENSOR_TABLE=beehive_readings
IOT_HIVE_COLUMN=device_id
IOT_TIMESTAMP_COLUMN=recorded_at
IOT_READING_AT_COLUMN=
IOT_TEMPERATURE_COLUMN=internal_temp
IOT_HUMIDITY_COLUMN=internal_humidity
IOT_CO2_COLUMN=internal_co2
IOT_WEIGHT_COLUMN=total_weight
IOT_EXTERNAL_TEMPERATURE_COLUMN=external_temp
IOT_EXTERNAL_HUMIDITY_COLUMN=external_humidity
IOT_BATTERY_VOLTAGE_COLUMN=battery_voltage

IOT_TIMESTAMPS_ARE_UTC=true
IOT_FEATURE_TIMEZONE=Asia/Colombo

IOT_TEMPERATURE_SCALE=1.0
IOT_HUMIDITY_SCALE=1.0
IOT_CO2_SCALE=1.0
IOT_WEIGHT_SCALE=1.0
```

### Password special characters

URL-encode characters such as `@`, `#`, `%`, `/`, `:` and `?` in the password. Example:

```powershell
python -c "from urllib.parse import quote; print(quote('YOUR_REAL_PASSWORD', safe=''))"
```

### Weight requirement

The historical model expects kilograms and the same physical meaning as the training hive-weight sensor. The screenshot shows `total_weight` around `4.8`. Keep `IOT_WEIGHT_SCALE=1.0` only when this is already kilograms.

- Database stores grams: `IOT_WEIGHT_SCALE=0.001`
- Database stores kilograms: `IOT_WEIGHT_SCALE=1.0`

A scale correction cannot fix a different physical quantity. If `total_weight` is only a small prototype-box component rather than total hive weight, the model output must remain provisional until local labelled validation is completed.

---

# 5. Configure `frontend/.env`

```powershell
Copy-Item `
  .\frontend\.env.live.example `
  .\frontend\.env `
  -Force
```

Contents:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:5000
VITE_IOT_REFRESH_MS=60000
```

---

# 6. Install dependencies

```powershell
cd .\backend\

pip install -e ".[dev]"
```

This installs `psycopg[binary]` for PostgreSQL/Supabase access.

---

# 7. Run code quality and tests

```powershell
ruff check . --fix
ruff check .

pytest `
  tests/modules/harvesting/test_live_hui_inference.py `
  tests/modules/harvesting/test_live_hui_monitor.py `
  tests/modules/harvesting/test_postgres_sensor_repository.py `
  -v
```

The package contains 13 focused tests.

Then:

```powershell
pytest -v
```

---

# 8. Verify all saved models

```powershell
python scripts/verify_live_hui_artifacts.py
```

Expected:

```json
{
  "status": "ok",
  "missing": []
}
```

When `.joblib` files are missing after cloning GitHub, retrieve them from the machine that trained the models or configure Git LFS. Never silently retrain during live inference.

---

# 9. Inspect PostgreSQL before inference

Run the included SQL in pgAdmin/Supabase:

```text
backend/sql/inspect_beehive_readings.sql
```

Then run:

```powershell
python scripts/check_live_hui_database.py
```

Review:

- `latest_timestamp`
- `latest_freshness_minutes`
- `history_rows_returned`
- `history_hives_returned`
- `optional_configured_columns_not_found`

### No rows returned

For genuine live use, keep:

```dotenv
IOT_HISTORY_REFERENCE=now
```

When the device is temporarily offline and you only need to test old rows, use:

```dotenv
IOT_HISTORY_REFERENCE=database_latest
```

The frontend will clearly display **historical replay mode**. Switch back to `now` before demonstrating live operation.

---

# 10. Check sensor compatibility with training data

```powershell
python scripts/profile_live_sensor_compatibility.py
```

This compares live sensor ranges with training-split ranges and writes:

```text
backend/artifacts/reports/harvesting/live_iot_sensor_compatibility.json
```

A `domain_shift_warning` is not a code failure. It means the sensor unit, calibration, hive construction or Sri Lankan environment differs from the historical dataset. Investigate especially `total_weight` before treating HUI as scientifically meaningful.

---

# 11. Run one complete live prediction

```powershell
python scripts/run_live_hui_once.py
```

A complete result includes:

```text
current_hui
current_class
predicted_hui_24h
predicted_hui_48h
predicted_hui_72h
hrsi
rate_of_change
recommended_window
sensor_status
```

### Insufficient live history

The model requires 192 contiguous hourly buckets:

- 168 hours for the classifier features
- 24 more hours for HUI-history features used by future-HUI regressors

Ten-minute readings are aggregated to hourly medians. A missing complete hour breaks the latest contiguous sequence; the API returns diagnostics instead of fabricating features.

---

# 12. Start the Flask API

```powershell
python app.py
```

Routes:

```text
GET  /api/harvesting/live-hui
GET  /api/harvesting/live-hui?hive_id=hive_01
GET  /api/harvesting/live-hui?refresh=true
POST /api/harvesting/live-hui/refresh
GET  /api/harvesting/live-hui/status
```

Test in another PowerShell window:

```powershell
Invoke-RestMethod `
  http://127.0.0.1:5000/api/harvesting/live-hui/status |
ConvertTo-Json -Depth 10
```

Force one refresh:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:5000/api/harvesting/live-hui/refresh `
  -ContentType "application/json" `
  -Body '{"hive_id":"hive_01"}' |
ConvertTo-Json -Depth 12
```

---

# 13. Start the React frontend

```powershell
cd ..\frontend\

npm run build
npm run dev
```

Open:

```text
http://localhost:5173/
```

Navigate to:

```text
Harvesting -> HUI Decision Support
```

The tab now displays live PostgreSQL values, current HUI and selected-model future HUI predictions.

---

# 14. Final verification checklist

- [ ] Latest database timestamp is recent
- [ ] At least 192 contiguous hourly buckets exist
- [ ] Weight is in kilograms and represents total hive weight
- [ ] Temperature is in degrees Celsius
- [ ] Humidity is percentage 0–100
- [ ] CO2 is ppm
- [ ] All saved model artifacts exist
- [ ] `run_live_hui_once.py` returns current and 24/48/72h HUI
- [ ] API status has no error
- [ ] Frontend shows `Live PostgreSQL IoT inference`
- [ ] Physical inspection remains required before harvesting

---

# 15. Commit only code, never secrets

```powershell
cd ..

git status
git add .
git restore --staged backend/.env frontend/.env 2>$null
git commit -m "Integrate live PostgreSQL IoT HUI inference"
git push
```

Confirm `.env` files are not tracked:

```powershell
git ls-files backend/.env frontend/.env
```

The command should print nothing.
