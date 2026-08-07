# Harvesting Research Gate and Future-Weight Forecasting

This package implements the next two stages.

## Stage A — Alert-policy research gate

The current refined policy alerts on approximately 98% of rows. It must not
be calibrated or displayed as HUI.

The gate accepts a policy only when it:

- detects both validation events;
- provides at least 12 hours median lead time;
- has precision at least twice validation prevalence;
- has precision no lower than the original policy;
- has no more false-positive rows than the original policy;
- has no larger alert occupancy than the original policy.

Run:

```powershell
python scripts/review_harvest_alert_policy.py
```

Output:

```text
artifacts/reports/harvesting/reviewed/alert_policy_gate/
├── research_gate_summary.json
├── policy_sweep_with_operational_metrics.csv
└── closest_non_deployable_policies.csv
```

Expected for the current results:

```text
status: no_research_safe_policy
ready_for_calibration: false
```

This is a valid research result. Preserve the classifier as a benchmark.

## Stage B — Label-independent future hive-weight forecasting

Run this stage when the gate finds no research-safe policy.

The forecasters predict future weight change at:

- 24 hours;
- 48 hours;
- 72 hours.

Models:

- persistence baseline;
- recent-trend baseline;
- Ridge regression;
- Random Forest;
- XGBoost;
- LightGBM.

Feature sets:

- weight only;
- no humidity.

The target endpoint must exist exactly in the same hive and official split.

Selection per horizon:

1. lowest validation MAE;
2. lowest validation median absolute error;
3. lowest absolute bias;
4. simpler model in a tie.

Run:

```powershell
python scripts/run_harvest_weight_forecasting.py
```

Outputs:

```text
artifacts/reports/harvesting/reviewed/weight_forecasting/
├── weight_forecasting_comparison.csv
├── weight_forecasting_summary.json
├── weight_forecasting_target_audit.json
├── selected_forecaster_per_hive_metrics.csv
├── selected_validation_predictions_24h.parquet
├── selected_validation_predictions_48h.parquet
├── selected_validation_predictions_72h.parquet
├── selected_test_predictions_24h.parquet
├── selected_test_predictions_48h.parquet
└── selected_test_predictions_72h.parquet
```

Models:

```text
artifacts/models/harvesting/weight_forecasting/
├── selected_weight_forecaster_24h.joblib
├── selected_weight_forecaster_48h.joblib
└── selected_weight_forecaster_72h.joblib
```

## Do not claim

Forecast accuracy does not prove honey maturity or optimal harvesting time.
The later readiness layer must remain transparent and provisional until
beekeeper-confirmed harvest and honey-quality records are available.
