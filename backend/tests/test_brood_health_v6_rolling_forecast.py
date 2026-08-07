from __future__ import annotations

import numpy as np
import pandas as pd

from multivari.modules.brood_health.analyzer import (
    classify_stability,
    classify_trend,
)
from multivari.modules.brood_health.features import (
    aggregate_live_hourly,
    map_iot_frame,
)
from multivari.modules.brood_health.forecast_indicators import (
    calibrate_forecast_stability_reference,
    forecast_bhsi,
    forecast_rod,
    interpolate_forecast_trajectory,
)


def _raw(last_timestamp: str, rows: int = 84) -> pd.DataFrame:
    end = pd.Timestamp(last_timestamp, tz="UTC")
    timestamps = pd.date_range(end=end, periods=rows, freq="10min")
    return pd.DataFrame(
        {
            "device_id": "hive_01",
            "recorded_at": timestamps,
            "internal_temp": 34.0 + np.sin(np.arange(rows) / 8.0),
            "internal_humidity": 65.0 + np.cos(np.arange(rows) / 9.0),
            "internal_co2": 850.0 + np.sin(np.arange(rows) / 6.0) * 50.0,
            "total_weight": 30.0 + np.arange(rows) * 0.001,
        }
    )


def test_rolling_hourly_anchor_moves_with_latest_reading() -> None:
    first = aggregate_live_hourly(map_iot_frame(_raw("2026-08-06 10:30")))
    second = aggregate_live_hourly(map_iot_frame(_raw("2026-08-06 10:40")))

    first_anchor = pd.Timestamp(first["timestamp"].max())
    second_anchor = pd.Timestamp(second["timestamp"].max())

    assert first_anchor == pd.Timestamp("2026-08-06 10:30", tz="UTC")
    assert second_anchor == pd.Timestamp("2026-08-06 10:40", tz="UTC")
    assert second_anchor - first_anchor == pd.Timedelta(minutes=10)
    assert first["timestamp"].sort_values().diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all()


def test_forecast_bhsi_and_rod_answer_different_questions() -> None:
    current = np.array([80.0, 80.0])
    actual = np.array(
        [
            [76.0, 72.0, 68.0, 64.0, 60.0, 56.0],  # smooth decline
            [79.0, 70.0, 78.0, 68.0, 76.0, 66.0],  # unstable decline
        ]
    )
    reference = calibrate_forecast_stability_reference(current, actual)
    bhsi = forecast_bhsi(current, actual, reference=reference)
    rod = forecast_rod(current, actual)

    assert bhsi[0] > bhsi[1]
    assert rod[0] < -0.5
    assert classify_trend(rod[0]) in {"Slow Declining", "Rapid Declining"}
    assert classify_stability(bhsi[0]) in {"Moderate", "High"}


def test_ten_minute_display_target_moves_with_forecast_anchor() -> None:
    hourly = [70, 68, 65, 62, 59, 56]
    first = interpolate_forecast_trajectory(
        current_score=72,
        hourly_scores=hourly,
        anchor_timestamp="2026-08-06T10:30:00+00:00",
        resolution_minutes=10,
    )
    second = interpolate_forecast_trajectory(
        current_score=72,
        hourly_scores=hourly,
        anchor_timestamp="2026-08-06T10:40:00+00:00",
        resolution_minutes=10,
    )

    assert len(first) == 37
    assert first[-1]["forecast_timestamp"] == "2026-08-06T16:30:00+00:00"
    assert second[-1]["forecast_timestamp"] == "2026-08-06T16:40:00+00:00"
    assert first[1]["value_kind"] == "display_interpolation"
    assert first[6]["value_kind"] == "native_hourly_model_output"
