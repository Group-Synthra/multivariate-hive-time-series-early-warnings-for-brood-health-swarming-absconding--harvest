from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd

try:
    import psycopg
    from psycopg import sql
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    psycopg = None
    sql = None


@dataclass(frozen=True)
class IotSettings:
    database_url: str
    sslmode: str = "require"
    schema: str = "public"
    table: str = "beehive_readings"
    hive_column: str = "device_id"
    timestamp_column: str = "recorded_at"
    temperature_column: str = "internal_temp"
    humidity_column: str = "internal_humidity"
    co2_column: str = "internal_co2"
    weight_column: str = "total_weight"
    external_temperature_column: str = "external_temp"
    external_humidity_column: str = "external_humidity"
    battery_voltage_column: str = "battery_voltage"
    hive_id: str | None = None
    timestamps_are_utc: bool = True
    feature_timezone: str = "Asia/Colombo"
    interval_minutes: int = 10
    history_hours: int = 192

    @classmethod
    def from_env(cls) -> IotSettings:
        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url:
            raise ValueError("DATABASE_URL is not configured in backend/.env.")
        return cls(
            database_url=database_url,
            sslmode=os.getenv("DATABASE_SSLMODE", "require").strip() or "require",
            schema=os.getenv("IOT_SCHEMA", "public").strip() or "public",
            table=os.getenv("IOT_SENSOR_TABLE", "beehive_readings").strip()
            or "beehive_readings",
            hive_column=os.getenv("IOT_HIVE_COLUMN", "device_id").strip() or "device_id",
            timestamp_column=(
                os.getenv("IOT_TIMESTAMP_COLUMN", "").strip()
                or os.getenv("IOT_READING_AT_COLUMN", "").strip()
                or "recorded_at"
            ),
            temperature_column=os.getenv("IOT_TEMPERATURE_COLUMN", "internal_temp").strip()
            or "internal_temp",
            humidity_column=os.getenv(
                "IOT_HUMIDITY_COLUMN", "internal_humidity"
            ).strip()
            or "internal_humidity",
            co2_column=os.getenv("IOT_CO2_COLUMN", "internal_co2").strip()
            or "internal_co2",
            weight_column=os.getenv("IOT_WEIGHT_COLUMN", "total_weight").strip()
            or "total_weight",
            external_temperature_column=os.getenv(
                "IOT_EXTERNAL_TEMPERATURE_COLUMN", "external_temp"
            ).strip()
            or "external_temp",
            external_humidity_column=os.getenv(
                "IOT_EXTERNAL_HUMIDITY_COLUMN", "external_humidity"
            ).strip()
            or "external_humidity",
            battery_voltage_column=os.getenv(
                "IOT_BATTERY_VOLTAGE_COLUMN", "battery_voltage"
            ).strip()
            or "battery_voltage",
            hive_id=os.getenv("IOT_HIVE_ID", "").strip() or None,
            timestamps_are_utc=_bool_env("IOT_TIMESTAMPS_ARE_UTC", True),
            feature_timezone=os.getenv("IOT_FEATURE_TIMEZONE", "Asia/Colombo").strip()
            or "Asia/Colombo",
            interval_minutes=max(1, int(os.getenv("IOT_INTERVAL_MINUTES", "10"))),
            history_hours=max(168, int(os.getenv("IOT_HISTORY_HOURS", "192"))),
        )

    @property
    def records_per_hive(self) -> int:
        per_hour = max(1, round(60 / self.interval_minutes))
        return int(self.history_hours * per_hour + per_hour * 2)

    @property
    def connection_url(self) -> str:
        parsed = urlparse(self.database_url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query.setdefault("sslmode", self.sslmode)
        return urlunparse(parsed._replace(query=urlencode(query)))

    @property
    def redacted_database(self) -> str:
        parsed = urlparse(self.database_url)
        host = parsed.hostname or "unknown-host"
        port = parsed.port or 5432
        database = (parsed.path or "/postgres").lstrip("/")
        return f"{parsed.scheme or 'postgresql'}://***:***@{host}:{port}/{database}"


class SupabaseIotRepository:
    def __init__(self, settings: IotSettings):
        self.settings = settings

    def fetch_latest(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        if psycopg is None or sql is None:
            raise RuntimeError(
                "PostgreSQL support is not installed. Run: python -m pip install -e \".[dev]\""
            )

        settings = self.settings
        with psycopg.connect(settings.connection_url, connect_timeout=15) as connection:
            columns = self._table_columns(connection)
            required = {
                settings.hive_column,
                settings.timestamp_column,
                settings.temperature_column,
                settings.humidity_column,
                settings.co2_column,
                settings.weight_column,
            }
            missing = sorted(required - columns)
            if missing:
                raise ValueError(
                    f"IoT table {settings.schema}.{settings.table} is missing columns {missing}. "
                    f"Available columns: {sorted(columns)}"
                )

            hive_id = settings.hive_id or self._latest_hive_id(connection)
            selected = [
                sql.SQL("{} AS recorded_at").format(sql.Identifier(settings.timestamp_column)),
                sql.SQL("{} AS device_id").format(sql.Identifier(settings.hive_column)),
                sql.SQL("{} AS internal_temp").format(
                    sql.Identifier(settings.temperature_column)
                ),
                sql.SQL("{} AS internal_humidity").format(
                    sql.Identifier(settings.humidity_column)
                ),
                sql.SQL("{} AS internal_co2").format(sql.Identifier(settings.co2_column)),
                sql.SQL("{} AS total_weight").format(sql.Identifier(settings.weight_column)),
            ]
            optional_columns = [
                (settings.external_temperature_column, "external_temp"),
                (settings.external_humidity_column, "external_humidity"),
                (settings.battery_voltage_column, "battery_voltage"),
            ]
            for source, alias in optional_columns:
                if source in columns:
                    selected.append(
                        sql.SQL("{} AS {}").format(sql.Identifier(source), sql.Identifier(alias))
                    )

            query = sql.SQL(
                "SELECT {fields} FROM {schema}.{table} "
                "WHERE {hive_column} = %s AND {timestamp_column} IS NOT NULL "
                "ORDER BY {timestamp_column} DESC LIMIT %s"
            ).format(
                fields=sql.SQL(", ").join(selected),
                schema=sql.Identifier(settings.schema),
                table=sql.Identifier(settings.table),
                hive_column=sql.Identifier(settings.hive_column),
                timestamp_column=sql.Identifier(settings.timestamp_column),
            )
            with connection.cursor() as cursor:
                cursor.execute(query, (hive_id, settings.records_per_hive))
                names = [description.name for description in cursor.description]
                rows = cursor.fetchall()

        frame = pd.DataFrame(rows, columns=names)
        frame = frame.iloc[::-1].reset_index(drop=True)
        metadata = {
            "source": "supabase_postgres",
            "database": settings.redacted_database,
            "schema": settings.schema,
            "table": settings.table,
            "hive_id": str(hive_id),
            "records_requested": settings.records_per_hive,
            "records_received": len(frame),
            "sampling_interval_minutes": settings.interval_minutes,
            "feature_timezone": settings.feature_timezone,
        }
        return frame, metadata

    def _table_columns(self, connection) -> set[str]:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                """,
                (self.settings.schema, self.settings.table),
            )
            return {row[0] for row in cursor.fetchall()}

    def _latest_hive_id(self, connection) -> str:
        query = sql.SQL(
            "SELECT {hive} FROM {schema}.{table} "
            "WHERE {hive} IS NOT NULL AND {timestamp} IS NOT NULL "
            "ORDER BY {timestamp} DESC LIMIT 1"
        ).format(
            hive=sql.Identifier(self.settings.hive_column),
            schema=sql.Identifier(self.settings.schema),
            table=sql.Identifier(self.settings.table),
            timestamp=sql.Identifier(self.settings.timestamp_column),
        )
        with connection.cursor() as cursor:
            cursor.execute(query)
            row = cursor.fetchone()
        if row is None:
            raise ValueError(
                f"IoT table {self.settings.schema}.{self.settings.table} contains no readings."
            )
        return str(row[0])


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
