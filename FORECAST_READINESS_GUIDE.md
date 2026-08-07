# Forecasting Research Gate and Provisional Readiness

This milestone starts only after future hive-weight forecasting has completed.

## Stage 1 — Forecasting research gate

The readiness prototype is blocked unless forecasting adds value beyond the
persistence baseline.

For each horizon, the gate checks:

- validation MAE improvement over persistence;
- test-to-validation MAE ratio;
- 72-hour performance not worse than persistence.

Default gate:

- at least 2 of 3 horizons improve validation MAE by at least 2%;
- the 72-hour selected model is not worse than persistence;
- test MAE is no more than twice validation MAE.

Run:

```powershell
python scripts/review_harvest_weight_forecasting.py
```

Output:

```text
artifacts/reports/harvesting/reviewed/forecast_readiness/
└── forecasting_research_gate.json
```

## Stage 2 — Provisional readiness prototype

Run only when:

```text
ready_for_readiness_prototype: true
```

The score is transparent and label-independent. It uses:

- recent 72-hour accumulation;
- distance from the recent 168-hour maximum;
- low predicted future weight-change rate;
- predicted slowdown relative to recent accumulation;
- agreement among 24-, 48- and 72-hour forecasts;
- temperature and CO2 stability;
- CO2 flatline quality penalty.

Component weights are prespecified in YAML and sum to one.

All normalization bounds and readiness-class thresholds are derived from the
official training split only.

The score is called:

```text
Provisional Harvest Readiness Score
```

It must not be presented as a calibrated probability.

## Derived indicators

- `HRSI`: trailing stability of the provisional readiness score;
- `HRRoC`: score change per hour over the configured lookback;
- candidate window: earliest 24/48/72-hour horizon with a plateau-like
  predicted rate, available only for Ready or High Priority rows.

These remain prototype indicators.

## Outputs

```text
artifacts/reports/harvesting/reviewed/forecast_readiness/
├── forecasting_research_gate.json
├── provisional_readiness_scores.parquet
├── latest_provisional_readiness_by_hive.csv
├── readiness_distribution_by_split.csv
├── readiness_normalization_parameters.json
├── readiness_thresholds.json
└── provisional_readiness_summary.json
```

Metadata:

```text
artifacts/models/harvesting/weight_forecasting/
└── provisional_readiness_metadata.json
```

## Research restriction

No operational deployment is allowed from this milestone. Prospective
beekeeper-confirmed harvest records and honey-maturity observations are still
required.
