# Continuous-History Feature Patch

The first feature build removed 32 positive rows because rolling
windows were calculated on the already filtered modelling table.
Every split-boundary or post-event purge interval was therefore
treated as a real sensor outage.

This patch calculates features from:

`data/processed/common_clean.parquet`

and then joins them to:

`data/processed/harvest_reviewed_72h_dataset.parquet`

Purged rows remain excluded as model samples, but they may contribute
past sensor history. This matches live inference and does not use
future values or labels.

Replace the two files in this patch, then run:

```powershell
ruff check . --fix
ruff check .
pytest -v
python scripts/build_reviewed_harvest_features.py
```

Expected test total after the patch: 27 passed.

The rebuilt audit should normally report:

- `feature_count`: 63
- `leakage_columns_present`: []
- `positive_rows_removed`: 0
- `output_positive_rows`: 864

Do not run reviewed feature EDA until the rebuilt audit has been
checked.
