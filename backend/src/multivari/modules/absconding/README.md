# Absconding early-warning module

This module replaces the previous standalone Absconding implementation with a pipeline that follows the new MULTIVARI shared data contract.

## What the module does

1. Loads `data/processed/common_clean.parquet` and the common chronological split manifest.
2. Converts `absconding_happened_1` into a leakage-safe **event within the next 72 hours** target.
3. Builds current/past-only temporal, lag, change, rolling, slope, persistence and multisensor-instability features.
4. Removes common split-boundary gaps before fitting.
5. Compares:
   - prior-probability baseline,
   - balanced logistic regression,
   - balanced random forest,
   - balanced Extra Trees,
   - Isolation Forest anomaly baseline.
6. Selects the model using validation PR-AUC, F2, precision and event-level recall.
7. Freezes the validation-selected threshold before test evaluation.
8. Saves a complete model bundle, metrics, event detection results, feature importance, plots and per-hive risk timelines.
9. Exposes summary, hive, image and live-inference API endpoints.

## Important scientific limitation

The common workbook contains only seven raw Absconding event-marker rows. The 72-hour future target creates additional warning-window rows, but those rows are correlated around a small number of distinct events. Therefore the supervised results are exploratory. The dashboard reports both row-level metrics and event-level detection/lead-time metrics so this limitation is visible.

## Run

From `backend/`:

```powershell
python scripts/run_absconding_pipeline.py
```

Fast code check with only two candidates:

```powershell
python scripts/run_absconding_pipeline.py --models logistic_balanced isolation_forest
```

Generated outputs:

- `artifacts/models/absconding/absconding_model_bundle.joblib`
- `artifacts/metrics/absconding/model_comparison.json`
- `artifacts/metrics/absconding/model_comparison.csv`
- `artifacts/metrics/absconding/feature_importance.csv`
- `artifacts/metrics/absconding/test_event_detection.csv`
- `artifacts/reports/absconding/absconding_dashboard.json`
- `artifacts/reports/absconding/*.png`
- `artifacts/predictions/absconding/latest_risk_per_hive.csv`
- `artifacts/predictions/absconding/absconding_risk_timeline.parquet`

## API

After starting `python app.py`:

- `GET /api/absconding/summary`
- `GET /api/absconding/metrics`
- `GET /api/absconding/hives`
- `GET /api/absconding/hives/<hive_id>`
- `POST /api/absconding/predict`
- `GET /api/absconding/images/<filename>`

Live prediction accepts:

```json
{
  "readings": [
    {
      "timestamp": "2026-08-04T10:00:00Z",
      "hive_id": "hive-live-01",
      "temperature_c": 34.2,
      "humidity_pct": 62.1,
      "co2_ppm": 840,
      "weight_kg": 31.8
    }
  ]
}
```

At least 168 hourly observations are required for each hive. Ten-minute readings may be supplied; the service aggregates them to hourly means before applying the saved training pipeline.

## Runtime and sampling

Candidate models use a capped, reproducible training sample that retains every positive warning row. The selected model is refit on a larger event-preserving sample configured by `final_training_rows`; this avoids treating hundreds of thousands of highly overlapping hourly rows as independent observations while keeping the pipeline practical on a development machine.
