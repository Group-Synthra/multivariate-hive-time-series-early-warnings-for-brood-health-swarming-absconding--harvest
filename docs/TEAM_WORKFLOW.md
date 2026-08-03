# Team workflow

## Shared team stage

1. Place the original workbook in `backend/data/raw/` locally. Do not modify it.
2. Run the common validation, cleaning, common EDA, and split-manifest pipeline.
3. Review the generated validation JSON, class-balance report, hive coverage, and sensor summaries.
4. Commit code, configuration, tests, documentation, and small manifests. Do not commit raw data, generated models, secrets, or `node_modules`.

## Module stage

Each member creates code only inside the assigned module package:

- `modules/brood_health/`
- `modules/swarming/`
- `modules/absconding/`
- `modules/harvesting/`

Each module must:

1. Load `common_clean.parquet` and `common_split_manifest.parquet`.
2. Define its prediction horizon and target transformation.
3. Build module-specific features using the shared feature functions.
4. Remove split-boundary rows before training.
5. Fit preprocessing only on the training split.
6. Evaluate on validation and test without random row shuffling.
7. Save the model together with feature names, units, frequency, lookback, horizon, and preprocessing configuration.
8. Reuse that exact pipeline for live IoT inference.

## Important target notes

- The supplied brood label is binary. A continuous brood score is a separate research construct and must not be treated as the same supervised target without justification.
- Swarming, absconding, and harvest labels mark rare events. Members should create future-event targets within a stated horizon.
- With only seven positive absconding rows, reliable supervised deep learning is not supported by this dataset alone. Treat results as exploratory or collect additional labelled events.
