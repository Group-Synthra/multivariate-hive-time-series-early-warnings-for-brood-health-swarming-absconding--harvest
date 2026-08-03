# Shared data contract

All modules must consume the canonical cleaned table created by the common pipeline.

## Canonical historical columns

- `timestamp`: timezone-naive historical hourly timestamp
- `hive_id`: stable hive identifier
- `temperature_c`
- `co2_ppm`
- `humidity_pct`
- `weight_kg`
- Four original label columns

## Live IoT feature parity

The live database may contain external temperature and humidity, but the supplied historical workbook does not. A deployed model must receive exactly the feature set used during training. Therefore:

1. Do not add external temperature or humidity to inference for a model trained without them.
2. Either keep those values as dashboard context, or enrich the historical dataset with matched external weather and retrain.
3. Aggregate 10-minute live readings to hourly values before using models trained on the hourly historical dataset.
4. Persist the preprocessing and feature configuration with every saved model.
