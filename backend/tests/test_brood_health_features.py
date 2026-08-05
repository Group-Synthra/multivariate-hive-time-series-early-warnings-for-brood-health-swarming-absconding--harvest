import numpy as np
import pandas as pd

from multivari.modules.brood_health.features import (
    TARGET_COLUMN,
    aggregate_live_hourly,
    build_feature_frame,
    build_supervised_dataset,
    map_iot_frame,
)


def historical_frame(hours=100):
    timestamp = pd.date_range("2026-01-01", periods=hours, freq="h")
    return pd.DataFrame({
        "hive_id": "hive-1",
        "timestamp": timestamp,
        "temperature_c": np.linspace(31, 36, hours),
        "humidity_pct": np.linspace(70, 62, hours),
        "co2_ppm": np.linspace(5000, 2500, hours),
        "weight_kg": np.linspace(28, 29, hours),
        TARGET_COLUMN: ([0] * 30) + ([1] * (hours - 30)),
    })


def test_target_is_never_an_input_feature():
    x, y, metadata, columns = build_supervised_dataset(historical_frame(), horizon_hours=6)
    assert TARGET_COLUMN not in columns
    assert TARGET_COLUMN not in x.columns
    assert len(x) == len(y) == len(metadata)
    assert (metadata["target_timestamp"] > metadata["timestamp"]).all()


def test_future_sensor_change_does_not_change_earlier_features():
    base = historical_frame()
    changed = base.copy()
    changed.loc[80:, "temperature_c"] = 999
    before = build_feature_frame(base).iloc[70]
    after = build_feature_frame(changed).iloc[70]
    pd.testing.assert_series_equal(before, after)


def test_live_mapping_and_hourly_aggregation():
    raw = pd.DataFrame({
        "device_id": ["d1"] * 12,
        "recorded_at": pd.date_range("2026-01-01", periods=12, freq="10min", tz="UTC"),
        "internal_temp": np.arange(12) + 30,
        "internal_humidity": np.arange(12) + 60,
        "internal_co2": np.arange(12) * 100 + 2000,
        "total_weight": np.arange(12) / 10 + 25,
        "external_temp": 29,
        "external_humidity": 75,
        "battery_voltage": 3.9,
    })
    hourly = aggregate_live_hourly(map_iot_frame(raw))
    assert list(hourly["raw_reading_count"]) == [6, 6]
    required_columns = {
        "hive_id",
        "timestamp",
        "temperature_c",
        "humidity_pct",
        "co2_ppm",
        "weight_kg",
    }
    assert required_columns.issubset(hourly)