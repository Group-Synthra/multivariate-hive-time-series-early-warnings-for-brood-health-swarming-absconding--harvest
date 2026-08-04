# Absconding early-warning module

This module uses a **separate labelled historical Absconding dataset** while the Brood Health,
Swarming, Harvesting and shared EDA pipelines continue using `common_clean.parquet`.

## Data isolation and integration

The supplied `hive_data_with_features.csv` is transformed into the same canonical sensor
contract used by live IoT inference:

- `timestamp`
- `hive_id`
- `temperature_c`
- `humidity_pct`
- `co2_ppm`
- `weight_kg`
- `external_temperature_c`
- `external_humidity_pct`

The source `absconding_event_label` is treated as an active event interval. The data pipeline
converts only each `0 -> 1` transition into `absconding_happened_1`, so the model predicts an
event onset instead of learning to recognise an event that is already in progress.

Generated module-specific data files:

- `data/processed/absconding_clean.parquet`
- `data/manifests/absconding_split_manifest.parquet`
- `artifacts/reports/absconding/absconding_data_profile.json`

These files do not replace or modify the shared common files.

## Supplied dataset profile

The uploaded file contains:

- 277,376 hourly records
- 48 hives
- 718 active-event rows
- 89 event-onset markers
- 58 independent episodes after merging onsets within 24 hours
- no missing values in the six training/live sensor fields

The model uses a leakage-safe **event within the next 24 hours** target, plus 1, 6, 24, 72
and 168-hour lag, change, rolling, stability and stress features. LSTM comparison uses
72-observation windows with stride 3.

## Run

Copy the source file to:

```text
data/raw/absconding/hive_data_with_features.csv
```

Then, from `backend/`:

```powershell
python scripts/run_absconding_data_pipeline.py
python scripts/run_absconding_pipeline.py
```

The model script can also prepare the data in one command:

```powershell
python scripts/run_absconding_pipeline.py --input "data/raw/absconding/hive_data_with_features.csv"
```

Train the optional LSTM after the classical pipeline:

```powershell
python scripts/run_absconding_lstm.py --epochs 30 --sequence-length 72 --stride 3
python scripts/run_absconding_pipeline.py
```

The second classical-pipeline run merges the saved LSTM metrics into the Model Training tab.

## Models

- prior-probability baseline
- rule-based environmental-stress baseline
- Gaussian Naive Bayes
- balanced Logistic Regression
- Ridge Classifier
- Decision Tree
- Random Forest
- Extra Trees
- Isolation Forest
- optional TensorFlow/Keras LSTM sequence model

Model and threshold selection use validation data only. The frozen threshold is then evaluated
on the chronological test split using row-level and episode-level metrics.

## Live IoT

The saved tabular model is used by the Supabase polling service. Ten-minute IoT readings are
mapped to the canonical schema, aggregated hourly and passed through the same feature builder.
External temperature and humidity are used when available. Missing optional external readings
are handled by the saved model's imputation pipeline.

The live endpoint requires at least 168 hourly observations for a hive and never invents a
probability before the required history exists.

## Research limitation

The separate historical dataset is much stronger than the original seven-marker common file,
but live Sri Lankan outputs still require local labels, calibration and biological validation
before production decisions are made solely from the model.
