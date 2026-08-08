from __future__ import annotations

from pathlib import Path

import pandas as pd

from multivari.iot.postgres_repository import PostgresSensorSettings
from multivari.modules.harvesting.live_hui_inference import (
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
        reading_at_column=None,
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
    engine.training_sensor_profile = {
        "weight_kg": {"q01": 31.4, "q99": 67.7},
        "internal_temperature_c": {"q01": 18.0, "q99": 37.2},
        "co2_ppm": {"q01": 355.0, "q99": 15242.0},
    }
    return engine


def test_old_model_ready_row_is_not_returned_as_live_prediction() -> None:
    predicted = pd.DataFrame(
        {
            "hive_id": ["hive_01"],
            "timestamp": [pd.Timestamp("2026-08-03 01:00:00")],
            "predicted_hui_24h": [30.0],
            "predicted_hui_48h": [31.0],
            "predicted_hui_72h": [32.0],
        }
    )
    diagnostics = [
        {
            "hive_id": "hive_01",
            "latest_hour": "2026-08-06T21:00:00",
            "latest_contiguous_hourly_rows": 84,
            "ready_for_full_hui": False,
        }
    ]

    aligned, updated = LiveHuiInferenceEngine._latest_aligned_ready_rows(predicted, diagnostics)

    assert aligned.empty
    assert updated[0]["live_prediction_ready"] is False
    assert updated[0]["prediction_timestamp_matches_latest_hour"] is False
    assert updated[0]["latest_model_lag_hours"] == 92.0


def test_exact_latest_model_row_is_accepted_when_history_is_ready() -> None:
    predicted = pd.DataFrame(
        {
            "hive_id": ["hive_01"],
            "timestamp": [pd.Timestamp("2026-08-06 21:00:00")],
            "predicted_hui_24h": [30.0],
            "predicted_hui_48h": [31.0],
            "predicted_hui_72h": [32.0],
        }
    )
    diagnostics = [
        {
            "hive_id": "hive_01",
            "latest_hour": "2026-08-06T21:00:00",
            "latest_contiguous_hourly_rows": 192,
            "ready_for_full_hui": True,
        }
    ]

    aligned, updated = LiveHuiInferenceEngine._latest_aligned_ready_rows(predicted, diagnostics)

    assert len(aligned) == 1
    assert updated[0]["live_prediction_ready"] is True
    assert updated[0]["prediction_timestamp_matches_latest_hour"] is True
    assert updated[0]["latest_model_lag_hours"] == 0.0


def test_weight_domain_shift_is_detected() -> None:
    engine = _engine_without_artifacts()
    checks, shifted = engine._sensor_domain_checks(
        pd.Series(
            {
                "weight_kg": 4.8,
                "temperature_c": 31.0,
                "co2_ppm": 742.0,
            }
        )
    )

    assert shifted is True
    weight = next(check for check in checks if check["sensor"] == "weight_kg")
    assert weight["outside_training_range"] is True


def test_weight_domain_shift_caps_confidence_at_low() -> None:
    engine = _engine_without_artifacts()
    score, label = engine._confidence(
        hrsi=100.0,
        completeness=100.0,
        freshness_minutes=0.0,
        domain_shift=True,
    )

    assert score == 49.9
    assert label == "Low"
