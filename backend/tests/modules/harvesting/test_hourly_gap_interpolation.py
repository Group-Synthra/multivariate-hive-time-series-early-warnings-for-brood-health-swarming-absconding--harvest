from __future__ import annotations

import pandas as pd

from multivari.modules.harvesting.hourly_gap_interpolation import (
    interpolate_bounded_hourly_gaps,
)

SENSORS = [
    "weight_kg",
    "temperature_c",
    "humidity_pct",
    "co2_ppm",
]
REQUIRED = SENSORS.copy()


def _frame(gap_hours: int) -> pd.DataFrame:
    first = pd.Timestamp("2026-08-03 01:00:00")
    second = first + pd.Timedelta(hours=gap_hours + 1)
    return pd.DataFrame(
        {
            "hive_id": ["hive_01", "hive_01"],
            "timestamp": [first, second],
            "weight_kg": [4.0, 4.9],
            "temperature_c": [30.0, 31.8],
            "humidity_pct": [78.0, 82.5],
            "co2_ppm": [650.0, 740.0],
            "split": ["live", "live"],
            "_live_target_placeholder": [0, 0],
            "readings_in_hour": [6, 6],
        }
    )


def test_interpolates_exactly_eight_missing_hours() -> None:
    output, summaries = interpolate_bounded_hourly_gaps(
        _frame(8),
        hive_column="hive_id",
        timestamp_column="timestamp",
        sensor_columns=SENSORS,
        required_sensor_columns=REQUIRED,
        max_gap_hours=8,
    )

    assert len(output) == 10
    assert int(output["is_imputed_hour"].sum()) == 8
    assert output[REQUIRED].notna().all(axis=None)
    assert summaries[0]["imputed_hourly_rows"] == 8


def test_does_not_partially_fill_gap_longer_than_limit() -> None:
    output, summaries = interpolate_bounded_hourly_gaps(
        _frame(9),
        hive_column="hive_id",
        timestamp_column="timestamp",
        sensor_columns=SENSORS,
        required_sensor_columns=REQUIRED,
        max_gap_hours=8,
    )

    assert len(output) == 2
    assert int(output["is_imputed_hour"].sum()) == 0
    assert summaries[0]["rejected_gap_count"] == 1


def test_original_sensor_rows_are_never_overwritten() -> None:
    source = _frame(8)
    output, _ = interpolate_bounded_hourly_gaps(
        source,
        hive_column="hive_id",
        timestamp_column="timestamp",
        sensor_columns=SENSORS,
        required_sensor_columns=REQUIRED,
        max_gap_hours=8,
    )

    observed = output.loc[~output["is_imputed_hour"]]
    assert observed["weight_kg"].tolist() == [4.0, 4.9]
    assert observed["readings_in_hour"].tolist() == [6, 6]
