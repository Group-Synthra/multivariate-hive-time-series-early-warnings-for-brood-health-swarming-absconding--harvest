from multivari.modules.brood_health.config import IoTSettings


def test_existing_env_variable_names_are_supported(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("IOT_HIVE_COLUMN", "device_id")
    monkeypatch.setenv("IOT_TEMPERATURE_COLUMN", "internal_temp")
    monkeypatch.setenv("IOT_HUMIDITY_COLUMN", "internal_humidity")
    monkeypatch.setenv("IOT_CO2_COLUMN", "internal_co2")
    monkeypatch.setenv("IOT_EXTERNAL_TEMPERATURE_COLUMN", "external_temp")
    monkeypatch.setenv("IOT_BATTERY_VOLTAGE_COLUMN", "battery_voltage")

    settings = IoTSettings.from_environment()

    assert settings.device_id_column == "device_id"
    assert settings.internal_temp_column == "internal_temp"
    assert settings.internal_humidity_column == "internal_humidity"
    assert settings.internal_co2_column == "internal_co2"
    assert settings.external_temp_column == "external_temp"
    assert settings.battery_column == "battery_voltage"
