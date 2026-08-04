from __future__ import annotations

import json

import pandas as pd

from multivari.common.schema import HIVE_COLUMN, TIMESTAMP_COLUMN
from multivari.modules.absconding.iot import IotSettings
from multivari.modules.absconding.iot_monitor import AbscondingIotMonitor
from multivari.modules.absconding.service import AbscondingService


def test_iot_settings_map_environment_without_exposing_password(monkeypatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://reader:secret-password@example.supabase.com:5432/postgres",
    )
    monkeypatch.setenv("IOT_HIVE_COLUMN", "device_id")
    monkeypatch.setenv("IOT_TIMESTAMP_COLUMN", "recorded_at")
    monkeypatch.setenv("IOT_FEATURE_TIMEZONE", "Asia/Colombo")
    settings = IotSettings.from_env()
    assert settings.hive_column == "device_id"
    assert settings.timestamp_column == "recorded_at"
    assert settings.feature_timezone == "Asia/Colombo"
    assert "secret-password" not in settings.redacted_database
    assert "sslmode=require" in settings.connection_url


def test_live_readings_are_mapped_and_hourly_aggregated(tmp_path) -> None:
    service = AbscondingService(tmp_path)
    raw = pd.DataFrame(
        {
            "recorded_at": ["2026-08-04T00:00:00Z", "2026-08-04T00:10:00Z"],
            "device_id": ["hive-01", "hive-01"],
            "internal_temp": [34.0, 36.0],
            "internal_humidity": [60.0, 62.0],
            "internal_co2": [800.0, 1000.0],
            "total_weight": [30.0, 31.0],
        }
    )
    normalised = service._normalise_readings(
        raw,
        feature_timezone="Asia/Colombo",
        timestamps_are_utc=True,
    )
    hourly = service._aggregate_hourly(normalised)
    assert list(normalised[HIVE_COLUMN].unique()) == ["hive-01"]
    assert normalised[TIMESTAMP_COLUMN].iloc[0].hour == 5
    assert len(hourly) == 1
    assert hourly["temperature_c"].iloc[0] == 35.0
    assert hourly["weight_kg"].iloc[0] == 30.5


def test_iot_monitor_persists_backend_prediction_cache(tmp_path) -> None:
    cache = tmp_path / "iot_live_latest.json"
    monitor = AbscondingIotMonitor(
        prediction_factory=lambda: {"status": "ok", "risk_percentage": 42.0},
        cache_path=cache,
        interval_minutes=10,
    )
    result = monitor.run_once()
    cached = monitor.read_cached()
    assert result["risk_percentage"] == 42.0
    assert json.loads(cache.read_text(encoding="utf-8"))["status"] == "ok"
    assert cached is not None
    assert cached["api_delivery_mode"] == "backend_cached_real_iot"
    assert cached["backend_iot_monitor"]["success_count"] == 1
