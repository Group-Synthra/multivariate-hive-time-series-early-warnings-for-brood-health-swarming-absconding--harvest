# Live HUI latest-hour and domain-shift safety patch

## Why this patch is required

The first live inference returned a model prediction timestamp of **2026-08-03 01:00** while the latest PostgreSQL sensor timestamp was **2026-08-06 16:12 UTC**. The newest contiguous hourly segment contained only 84 hours, below the 192 hours required for current plus future HUI. The earlier implementation selected an older model-ready row and combined it with the latest sensor status.

This patch prevents old model rows from being presented as current live predictions. It also detects the large live-weight domain shift and caps evidence confidence at Low when the current hive weight is outside the historical training range.

## Changes

- Current HUI readiness requires 168 latest contiguous complete hours.
- Full current plus 24/48/72-hour HUI requires 192 latest contiguous complete hours.
- A prediction is returned only when its model timestamp equals the newest hourly IoT bucket.
- Old model-ready rows are retained only in diagnostics and are never labelled live.
- The latest continuous segment is used for the chart.
- Training q01-q99 sensor ranges are loaded from the frozen training feature dataset.
- Live weight outside the historical range creates a domain-shift warning.
- Domain shift caps prototype evidence confidence at 49.9 / Low.
- The frontend displays current-HUI readiness, future-HUI readiness, model lag, and domain shift.
- A new hourly-gap report identifies exactly where data gaps occurred.

## Installation

Extract the ZIP into the project root and run:

```powershell
python .\apply_live_hui_latest_safety_patch.py
```

Then from `backend`:

```powershell
ruff check . --fix
ruff check .

pytest `
  tests/modules/harvesting/test_live_hui_inference.py `
  tests/modules/harvesting/test_live_hui_monitor.py `
  tests/modules/harvesting/test_postgres_sensor_repository.py `
  tests/modules/harvesting/test_live_hui_safety.py `
  -v

pytest -v
```

The four new safety tests should pass. The full total should increase from 97 to 101 tests.

## Expected first inference after patch

With only 84 latest contiguous hourly buckets, this command should no longer return the stale 3 August prediction:

```powershell
python scripts/run_live_hui_once.py
```

It should report that the newest IoT hour does not yet have 192 contiguous complete hourly observations. This is the correct result.

Run the gap report:

```powershell
python scripts/report_live_hui_hourly_gaps.py
```

For the current 84-hour segment:

- approximately 84 more contiguous hours are needed for current HUI (168 total),
- approximately 108 more contiguous hours are needed for current plus future HUI (192 total),
- any new missing hour resets the latest-contiguous counter.

## Weight-domain interpretation

The live value of approximately 4.8 kg uses the correct kilogram unit, but it remains outside the historical training range of approximately 31.4-67.7 kg. This is a domain-transfer problem, not a unit-conversion problem.

Keep:

```dotenv
IOT_WEIGHT_SCALE=1.0
```

The live model may still be demonstrated as an experimental transfer output, but the HUI must not be presented as validated for Sri Lankan hives until local harvest labels and a local calibration dataset are collected.
