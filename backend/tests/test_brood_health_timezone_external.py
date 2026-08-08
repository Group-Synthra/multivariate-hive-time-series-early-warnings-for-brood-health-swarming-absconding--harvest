import numpy as np
import pandas as pd

from multivari.modules.brood_health.features import (
    HISTORICAL_FEATURE_TIMEZONE,
    build_feature_frame,
    map_iot_frame,
    normalise_historical,
)


def _base_frame(timestamp):
    return pd.DataFrame(
        {
            "hive_id": ["h1"],
            "timestamp": [timestamp],
            "temperature_c": [35.0],
            "humidity_pct": [65.0],
            "co2_ppm": [900.0],
            "weight_kg": [40.0],
            "external temperature": [30.0],
            "external humidity": [45.0],
        }
    )


def test_historical_naive_timestamp_preserves_arizona_local_clock() -> None:
    historical = normalise_historical(
        _base_frame("2024-01-01 08:00:00"),
        naive_timezone=HISTORICAL_FEATURE_TIMEZONE,
    )
    assert historical.loc[0, "timestamp"].hour == 15
    features = build_feature_frame(
        historical,
        feature_timezone=HISTORICAL_FEATURE_TIMEZONE,
    )
    np.testing.assert_allclose(features.loc[0, "hour_sin"], np.sin(2 * np.pi * 8 / 24))
    np.testing.assert_allclose(features.loc[0, "hour_cos"], np.cos(2 * np.pi * 8 / 24))


def test_live_utc_timestamp_uses_colombo_local_clock() -> None:
    raw = pd.DataFrame(
        {
            "device_id": ["h1"],
            "recorded_at": ["2024-01-01T02:30:00+00:00"],
            "internal_temp": [35.0],
            "internal_humidity": [65.0],
            "internal_co2": [900.0],
            "total_weight": [4.0],
            "external_temp": [30.0],
            "external_humidity": [45.0],
        }
    )
    live = map_iot_frame(
        raw,
        timestamps_are_utc=True,
        feature_timezone="Asia/Colombo",
    )
    features = build_feature_frame(live, feature_timezone="Asia/Colombo")
    np.testing.assert_allclose(features.loc[0, "hour_sin"], np.sin(2 * np.pi * 8 / 24))
    np.testing.assert_allclose(features.loc[0, "hour_cos"], np.cos(2 * np.pi * 8 / 24))


def test_external_parameters_enter_feature_frame() -> None:
    historical = normalise_historical(_base_frame("2024-01-01 08:00:00"))
    features = build_feature_frame(historical)
    assert features.loc[0, "external_temp"] == 30.0
    assert features.loc[0, "external_humidity"] == 45.0
    assert features.loc[0, "internal_external_temp_delta"] == 5.0
    assert features.loc[0, "internal_external_humidity_delta"] == 20.0
