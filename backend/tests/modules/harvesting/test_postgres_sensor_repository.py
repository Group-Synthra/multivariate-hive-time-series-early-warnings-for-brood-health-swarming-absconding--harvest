from __future__ import annotations

import pytest

from multivari.iot.postgres_repository import (
    LiveSensorConfigurationError,
    PostgresSensorRepository,
    PostgresSensorSettings,
)


def _settings() -> PostgresSensorSettings:
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


def test_optional_reading_at_is_ignored_when_table_does_not_contain_it() -> None:
    repository = PostgresSensorRepository(_settings())
    available = {
        "device_id",
        "recorded_at",
        "internal_temp",
        "internal_humidity",
        "internal_co2",
        "total_weight",
        "external_temp",
        "external_humidity",
        "battery_voltage",
    }

    resolved = repository._resolve_columns(available)

    assert resolved["reading_at"] is None
    assert resolved["timestamp"] == "recorded_at"
    assert resolved["battery_voltage"] == "battery_voltage"


def test_missing_required_sensor_column_is_rejected() -> None:
    repository = PostgresSensorRepository(_settings())
    available = {
        "device_id",
        "recorded_at",
        "internal_temp",
        "internal_humidity",
        "total_weight",
    }

    with pytest.raises(LiveSensorConfigurationError, match="internal_co2"):
        repository._resolve_columns(available)


def test_blank_reading_at_environment_is_supported(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("IOT_READING_AT_COLUMN", "")
    settings = PostgresSensorSettings.from_env()
    assert settings.reading_at_column is None


def test_history_shorter_than_192_hours_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("IOT_HISTORY_HOURS", "191")
    with pytest.raises(LiveSensorConfigurationError, match="at least 192"):
        PostgresSensorSettings.from_env()
