from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from multivari.iot.postgres_repository import PostgresSensorSettings
from multivari.modules.harvesting.live_hui_inference import (
    CURRENT_HUI_COLUMN,
    LiveHuiArtifactSettings,
    LiveHuiInferenceEngine,
)


def _sensor_settings() -> PostgresSensorSettings:
    return PostgresSensorSettings(
        database_url="postgresql://example",
        sslmode="require",
        schema="public",
        table="beehive_readings",
        hive_column="device_id",
        timestamp_column="recorded_at",
        reading_at_column="reading_at",
        temperature_column="internal_temp",
        humidity_column="internal_humidity",
        co2_column="internal_co2",
        weight_column="total_weight",
        external_temperature_column="external_temp",
        external_humidity_column="external_humidity",
        battery_voltage_column="battery_voltage",
        timestamps_are_utc=True,
        feature_timezone="Asia/Colombo",
        history_hours=336,
        history_reference="now",
        configured_hive_id=None,
        minimum_readings_per_hour=1,
        temperature_scale=1.0,
        humidity_scale=1.0,
        co2_scale=1.0,
        weight_scale=1.0,
    )


def _engine_without_artifacts() -> LiveHuiInferenceEngine:
    engine = object.__new__(LiveHuiInferenceEngine)
    engine.sensor_settings = _sensor_settings()
    engine.calibration_gate = {
        "gate_passed": False,
        "selected_method": "platt",
    }
    engine.artifact_settings = LiveHuiArtifactSettings(
        classifier_model_path=Path("classifier.joblib"),
        classifier_features_path=Path("features.json"),
        calibrator_path=Path("calibrator.joblib"),
        harvesting_config_path=Path("harvesting.yaml"),
        calibration_gate_path=Path("calibration_gate.json"),
        future_hui_gate_path=Path("hui_gate.json"),
        future_model_directory=Path("future_models"),
        stale_after_minutes=30,
        series_rows_per_hive=168,
    )
    return engine


def test_prepare_hourly_history_maps_sl_columns_and_timezone() -> None:
    engine = _engine_without_artifacts()
    timestamps = pd.date_range(
        "2026-07-01T00:00:00Z",
        periods=192 * 6,
        freq="10min",
    )
    raw = pd.DataFrame(
        {
            "source_hive_id": "SL-HIVE-01",
            "source_timestamp": timestamps,
            "source_recorded_at": timestamps,
            "source_reading_at": timestamps,
            "internal_temperature": 34.5,
            "internal_humidity": 62.0,
            "internal_co2": 780.0,
            "total_weight": 42.0,
            "external_temperature": 29.0,
            "external_humidity": 75.0,
            "battery_voltage": 3.9,
        }
    )

    hourly, latest, diagnostics = engine.prepare_hourly_history(raw)

    assert len(hourly) >= 192
    assert hourly["hive_id"].nunique() == 1
    assert hourly["timestamp"].iloc[0].hour == 5  # UTC midnight -> 05:30 SL, floored.
    assert float(hourly["weight_kg"].iloc[-1]) == 42.0
    assert len(latest) == 1
    assert diagnostics[0]["ready_for_full_hui"] is True


class _SimpleCalibrator:
    def predict(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float).reshape(-1)
        return np.clip(array * 0.5, 0.0, 1.0)


def test_apply_calibrator_supports_custom_platt_predict_method() -> None:
    result = LiveHuiInferenceEngine._apply_calibrator(
        _SimpleCalibrator(),
        np.array([0.2, 0.8]),
    )
    assert np.allclose(result, [0.1, 0.4])


def test_limited_calibration_caps_evidence_confidence_at_moderate() -> None:
    engine = _engine_without_artifacts()
    score, label = engine._confidence(
        hrsi=100.0,
        completeness=100.0,
        freshness_minutes=0.0,
    )
    assert score == 74.9
    assert label == "Moderate"


def test_stale_sensor_reduces_confidence() -> None:
    engine = _engine_without_artifacts()
    fresh_score, _ = engine._confidence(
        hrsi=70.0,
        completeness=100.0,
        freshness_minutes=5.0,
    )
    stale_score, _ = engine._confidence(
        hrsi=70.0,
        completeness=100.0,
        freshness_minutes=60.0,
    )
    assert stale_score < fresh_score


def test_recommended_window_uses_earliest_ready_horizon() -> None:
    row = pd.Series(
        {
            CURRENT_HUI_COLUMN: 35.0,
            "predicted_hui_24h": 52.0,
            "predicted_hui_48h": 64.0,
            "predicted_hui_72h": 70.0,
        }
    )
    window, _ = LiveHuiInferenceEngine._recommended_window(row)
    assert window == "Within 24–48 hours"


def test_rate_labels_use_frozen_slope_boundaries() -> None:
    assert LiveHuiInferenceEngine._rate_label(0.51) == "Increasing"
    assert LiveHuiInferenceEngine._rate_label(-0.51) == "Decreasing"
    assert LiveHuiInferenceEngine._rate_label(0.5) == "Stable"
