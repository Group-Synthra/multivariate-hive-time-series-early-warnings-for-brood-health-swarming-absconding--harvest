from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def backend_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_backend_environment(*, override: bool = False) -> Path:
    env_path = backend_root() / ".env"
    load_dotenv(dotenv_path=env_path, override=override)
    return env_path


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


@dataclass(frozen=True)
class BroodPaths:
    backend: Path = field(default_factory=backend_root)

    @property
    def clean_data(self) -> Path:
        return self.backend / "data" / "processed" / "common_clean.parquet"

    @property
    def raw_workbook(self) -> Path:
        return (
            self.backend
            / "data"
            / "raw"
            / "Common_Beehive_Complete_Training_Dataset_311044.xlsx"
        )

    @property
    def split_manifest(self) -> Path:
        return self.backend / "data" / "manifests" / "common_split_manifest.parquet"

    @property
    def model_dir(self) -> Path:
        return self.backend / "artifacts" / "models" / "brood_health"

    @property
    def metrics_dir(self) -> Path:
        return self.backend / "artifacts" / "metrics" / "brood_health"

    @property
    def report_dir(self) -> Path:
        return self.backend / "artifacts" / "reports" / "brood_health"

    @property
    def model_bundle(self) -> Path:
        return self.model_dir / "brood_health_forecaster_v6.joblib"

    @property
    def training_summary(self) -> Path:
        return self.metrics_dir / "training_summary_v6.json"

    @property
    def feature_importance(self) -> Path:
        return self.metrics_dir / "feature_importance_v6.csv"

    @property
    def test_predictions(self) -> Path:
        return self.metrics_dir / "test_predictions_v6.csv"

    @property
    def model_comparison(self) -> Path:
        return self.metrics_dir / "model_comparison_v6.csv"

    @property
    def weight_sensitivity(self) -> Path:
        return self.metrics_dir / "weight_sensitivity_v6.csv"

    @property
    def eda_cache(self) -> Path:
        return self.metrics_dir / "eda_cache_v6.json"


@dataclass(frozen=True)
class IoTSettings:
    database_url: str
    schema: str
    table: str
    device_id_column: str
    timestamp_column: str
    internal_temp_column: str
    internal_humidity_column: str
    internal_co2_column: str
    weight_column: str
    external_temp_column: str
    external_humidity_column: str
    battery_column: str
    reading_at_column: str
    lookback_hours: int
    minimum_hourly_rows: int
    refresh_seconds: int
    connect_timeout_seconds: int
    timestamps_are_utc: bool
    feature_timezone: str
    sslmode: str
    weight_scale_factor: float
    weight_offset_kg: float

    @classmethod
    def from_environment(cls) -> IoTSettings:
        load_backend_environment()
        return cls(
            database_url=_first_env("DATABASE_URL"),
            schema=_first_env("IOT_SCHEMA", default="public"),
            table=_first_env("IOT_SENSOR_TABLE", default="beehive_readings"),
            device_id_column=_first_env(
                "IOT_HIVE_COLUMN", "IOT_DEVICE_ID_COLUMN", default="device_id"
            ),
            timestamp_column=_first_env("IOT_TIMESTAMP_COLUMN", default="recorded_at"),
            internal_temp_column=_first_env(
                "IOT_TEMPERATURE_COLUMN",
                "IOT_INTERNAL_TEMP_COLUMN",
                default="internal_temp",
            ),
            internal_humidity_column=_first_env(
                "IOT_HUMIDITY_COLUMN",
                "IOT_INTERNAL_HUMIDITY_COLUMN",
                default="internal_humidity",
            ),
            internal_co2_column=_first_env(
                "IOT_CO2_COLUMN", "IOT_INTERNAL_CO2_COLUMN", default="internal_co2"
            ),
            weight_column=_first_env("IOT_WEIGHT_COLUMN", default="total_weight"),
            external_temp_column=_first_env(
                "IOT_EXTERNAL_TEMPERATURE_COLUMN",
                "IOT_EXTERNAL_TEMP_COLUMN",
                default="external_temp",
            ),
            external_humidity_column=_first_env(
                "IOT_EXTERNAL_HUMIDITY_COLUMN", default="external_humidity"
            ),
            battery_column=_first_env(
                "IOT_BATTERY_VOLTAGE_COLUMN",
                "IOT_BATTERY_COLUMN",
                default="battery_voltage",
            ),
            reading_at_column=_first_env(
                "IOT_READING_AT_COLUMN", default="reading_at"
            ),
            lookback_hours=max(
                24, int(_first_env("IOT_LOOKBACK_HOURS", default="168"))
            ),
            minimum_hourly_rows=max(
                6, int(_first_env("IOT_MIN_HOURLY_ROWS", default="72"))
            ),
            refresh_seconds=max(
                30, int(_first_env("IOT_REFRESH_SECONDS", default="600"))
            ),
            connect_timeout_seconds=max(
                2, int(_first_env("IOT_CONNECT_TIMEOUT_SECONDS", default="12"))
            ),
            timestamps_are_utc=_first_env(
                "IOT_TIMESTAMPS_ARE_UTC", default="true"
            ).lower()
            in {"1", "true", "yes", "on"},
            feature_timezone=_first_env(
                "IOT_FEATURE_TIMEZONE", default="Asia/Colombo"
            ),
            sslmode=_first_env("DATABASE_SSLMODE", default="require"),
            # Use 0.001 when the database stores grams; keep 1.0 for kilograms.
            weight_scale_factor=float(
                _first_env("IOT_WEIGHT_SCALE_FACTOR", default="1.0")
            ),
            # Optional tare correction after unit conversion.
            weight_offset_kg=float(
                _first_env("IOT_WEIGHT_OFFSET_KG", default="0.0")
            ),
        )


PATHS = BroodPaths()
