from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HISTORY_REFERENCE_VALUES = {"now", "database_latest"}


class LiveSensorConfigurationError(RuntimeError):
    """Raised when the PostgreSQL live-sensor configuration is incomplete."""


class LiveSensorDatabaseError(RuntimeError):
    """Raised when live sensor rows cannot be read from PostgreSQL."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else float(raw)


def _validate_identifier(value: str, *, setting_name: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise LiveSensorConfigurationError(
            f"{setting_name} must be a simple PostgreSQL identifier. "
            f"Received: {value!r}"
        )
    return value


@dataclass(frozen=True)
class PostgresSensorSettings:
    database_url: str
    sslmode: str
    schema: str
    table: str
    hive_column: str
    timestamp_column: str
    reading_at_column: str | None
    temperature_column: str
    humidity_column: str
    co2_column: str
    weight_column: str
    external_temperature_column: str | None
    external_humidity_column: str | None
    battery_voltage_column: str | None
    timestamps_are_utc: bool
    feature_timezone: str
    history_hours: int
    history_reference: str
    configured_hive_id: str | None
    minimum_readings_per_hour: int
    temperature_scale: float
    humidity_scale: float
    co2_scale: float
    weight_scale: float

    @classmethod
    def from_env(cls) -> PostgresSensorSettings:
        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url:
            raise LiveSensorConfigurationError(
                "DATABASE_URL is required for live HUI inference."
            )

        history_hours = int(os.getenv("IOT_HISTORY_HOURS", "336"))
        if history_hours < 192:
            raise LiveSensorConfigurationError(
                "IOT_HISTORY_HOURS must be at least 192. The pipeline needs "
                "168 hours for classifier features and 24 additional hours "
                "for HUI-history features."
            )

        history_reference = os.getenv("IOT_HISTORY_REFERENCE", "now").strip().lower()
        if history_reference not in _HISTORY_REFERENCE_VALUES:
            raise LiveSensorConfigurationError(
                "IOT_HISTORY_REFERENCE must be either 'now' or 'database_latest'."
            )

        minimum_readings_per_hour = int(
            os.getenv("IOT_MIN_READINGS_PER_HOUR", "1")
        )
        if minimum_readings_per_hour <= 0:
            raise LiveSensorConfigurationError(
                "IOT_MIN_READINGS_PER_HOUR must be a positive integer."
            )

        # The uploaded table contains recorded_at but does not show reading_at.
        # A blank value disables the optional reading_at column.
        reading_at = os.getenv("IOT_READING_AT_COLUMN", "").strip() or None
        external_temperature = (
            os.getenv("IOT_EXTERNAL_TEMPERATURE_COLUMN", "external_temp").strip()
            or None
        )
        external_humidity = (
            os.getenv("IOT_EXTERNAL_HUMIDITY_COLUMN", "external_humidity").strip()
            or None
        )
        battery_voltage = (
            os.getenv("IOT_BATTERY_VOLTAGE_COLUMN", "battery_voltage").strip()
            or None
        )

        identifier_values: dict[str, str | None] = {
            "IOT_SCHEMA": os.getenv("IOT_SCHEMA", "public").strip(),
            "IOT_SENSOR_TABLE": os.getenv(
                "IOT_SENSOR_TABLE", "beehive_readings"
            ).strip(),
            "IOT_HIVE_COLUMN": os.getenv("IOT_HIVE_COLUMN", "device_id").strip(),
            "IOT_TIMESTAMP_COLUMN": os.getenv(
                "IOT_TIMESTAMP_COLUMN", "recorded_at"
            ).strip(),
            "IOT_READING_AT_COLUMN": reading_at,
            "IOT_TEMPERATURE_COLUMN": os.getenv(
                "IOT_TEMPERATURE_COLUMN", "internal_temp"
            ).strip(),
            "IOT_HUMIDITY_COLUMN": os.getenv(
                "IOT_HUMIDITY_COLUMN", "internal_humidity"
            ).strip(),
            "IOT_CO2_COLUMN": os.getenv("IOT_CO2_COLUMN", "internal_co2").strip(),
            "IOT_WEIGHT_COLUMN": os.getenv(
                "IOT_WEIGHT_COLUMN", "total_weight"
            ).strip(),
            "IOT_EXTERNAL_TEMPERATURE_COLUMN": external_temperature,
            "IOT_EXTERNAL_HUMIDITY_COLUMN": external_humidity,
            "IOT_BATTERY_VOLTAGE_COLUMN": battery_voltage,
        }
        validated: dict[str, str | None] = {}
        for setting_name, value in identifier_values.items():
            validated[setting_name] = (
                None
                if value is None
                else _validate_identifier(value, setting_name=setting_name)
            )

        configured_hive = os.getenv("IOT_HIVE_ID", "").strip() or None

        return cls(
            database_url=database_url,
            sslmode=os.getenv("DATABASE_SSLMODE", "require").strip() or "require",
            schema=str(validated["IOT_SCHEMA"]),
            table=str(validated["IOT_SENSOR_TABLE"]),
            hive_column=str(validated["IOT_HIVE_COLUMN"]),
            timestamp_column=str(validated["IOT_TIMESTAMP_COLUMN"]),
            reading_at_column=validated["IOT_READING_AT_COLUMN"],
            temperature_column=str(validated["IOT_TEMPERATURE_COLUMN"]),
            humidity_column=str(validated["IOT_HUMIDITY_COLUMN"]),
            co2_column=str(validated["IOT_CO2_COLUMN"]),
            weight_column=str(validated["IOT_WEIGHT_COLUMN"]),
            external_temperature_column=validated[
                "IOT_EXTERNAL_TEMPERATURE_COLUMN"
            ],
            external_humidity_column=validated["IOT_EXTERNAL_HUMIDITY_COLUMN"],
            battery_voltage_column=validated["IOT_BATTERY_VOLTAGE_COLUMN"],
            timestamps_are_utc=_env_bool("IOT_TIMESTAMPS_ARE_UTC", True),
            feature_timezone=os.getenv(
                "IOT_FEATURE_TIMEZONE", "Asia/Colombo"
            ).strip()
            or "Asia/Colombo",
            history_hours=history_hours,
            history_reference=history_reference,
            configured_hive_id=configured_hive,
            minimum_readings_per_hour=minimum_readings_per_hour,
            temperature_scale=_env_float("IOT_TEMPERATURE_SCALE", 1.0),
            humidity_scale=_env_float("IOT_HUMIDITY_SCALE", 1.0),
            co2_scale=_env_float("IOT_CO2_SCALE", 1.0),
            weight_scale=_env_float("IOT_WEIGHT_SCALE", 1.0),
        )


class PostgresSensorRepository:
    """Read recent IoT sensor history from the configured PostgreSQL table."""

    def __init__(self, settings: PostgresSensorSettings):
        self.settings = settings

    @staticmethod
    def _driver():
        try:
            import psycopg
            from psycopg import sql
            from psycopg.rows import dict_row
        except ImportError as error:
            raise LiveSensorConfigurationError(
                "psycopg is not installed. Run: "
                "pip install 'psycopg[binary]>=3.2,<4.0'"
            ) from error
        return psycopg, sql, dict_row

    def _connect(self):
        psycopg, _, dict_row = self._driver()
        return psycopg.connect(
            self.settings.database_url,
            sslmode=self.settings.sslmode,
            connect_timeout=15,
            row_factory=dict_row,
            options="-c default_transaction_read_only=on",
        )

    def _table_columns(self, connection: Any) -> set[str]:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (self.settings.schema, self.settings.table),
            )
            rows = cursor.fetchall()
        columns = {str(row["column_name"]) for row in rows}
        if not columns:
            raise LiveSensorDatabaseError(
                f"Table {self.settings.schema}.{self.settings.table} was not found "
                "or the configured database user cannot inspect it."
            )
        return columns

    def _resolve_columns(self, available: set[str]) -> dict[str, str | None]:
        settings = self.settings
        required = {
            "hive": settings.hive_column,
            "timestamp": settings.timestamp_column,
            "temperature": settings.temperature_column,
            "humidity": settings.humidity_column,
            "co2": settings.co2_column,
            "weight": settings.weight_column,
        }
        missing = sorted(value for value in required.values() if value not in available)
        if missing:
            raise LiveSensorConfigurationError(
                "The PostgreSQL table is missing required configured columns: "
                f"{missing}. Available columns: {sorted(available)}"
            )

        def optional(column: str | None) -> str | None:
            return column if column and column in available else None

        return {
            **required,
            "reading_at": optional(settings.reading_at_column),
            "external_temperature": optional(settings.external_temperature_column),
            "external_humidity": optional(settings.external_humidity_column),
            "battery_voltage": optional(settings.battery_voltage_column),
        }

    @staticmethod
    def _timestamp_expression(sql: Any, resolved: dict[str, str | None]):
        reading_at = resolved["reading_at"]
        recorded_at = str(resolved["timestamp"])
        if reading_at:
            return sql.SQL("COALESCE({}, {})").format(
                sql.Identifier(str(reading_at)),
                sql.Identifier(recorded_at),
            )
        return sql.Identifier(recorded_at)

    def _reference_timestamp(
        self,
        connection: Any,
        *,
        resolved: dict[str, str | None],
        hive_id: str | None,
    ) -> datetime:
        if self.settings.history_reference == "now":
            return datetime.now(UTC)

        _, sql, _ = self._driver()
        timestamp_expression = self._timestamp_expression(sql, resolved)
        query = sql.SQL("SELECT MAX({}) AS reference_timestamp FROM {}.{}").format(
            timestamp_expression,
            sql.Identifier(self.settings.schema),
            sql.Identifier(self.settings.table),
        )
        parameters: list[Any] = []
        if hive_id:
            query += sql.SQL(" WHERE {} = %s").format(
                sql.Identifier(str(resolved["hive"]))
            )
            parameters.append(hive_id)

        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            row = cursor.fetchone()
        value = None if row is None else row.get("reference_timestamp")
        if value is None:
            raise LiveSensorDatabaseError(
                "No timestamp is available in the configured IoT table."
            )
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("UTC")
        else:
            parsed = parsed.tz_convert("UTC")
        return parsed.to_pydatetime()

    @staticmethod
    def _optional_selection(
        sql: Any,
        column: str | None,
        alias: str,
        *,
        null_type: str = "double precision",
    ):
        if column:
            return sql.SQL("{} AS {}").format(
                sql.Identifier(column),
                sql.Identifier(alias),
            )
        return sql.SQL(f"NULL::{null_type} AS {{}}").format(sql.Identifier(alias))

    def fetch_recent(self, *, hive_id: str | None = None) -> pd.DataFrame:
        _, sql, _ = self._driver()
        settings = self.settings
        selected_hive = hive_id or settings.configured_hive_id

        try:
            with self._connect() as connection:
                available = self._table_columns(connection)
                resolved = self._resolve_columns(available)
                reference_timestamp = self._reference_timestamp(
                    connection,
                    resolved=resolved,
                    hive_id=selected_hive,
                )
                start_utc = reference_timestamp - timedelta(hours=settings.history_hours)
                timestamp_expression = self._timestamp_expression(sql, resolved)

                selections: list[Any] = [
                    sql.SQL("{} AS source_hive_id").format(
                        sql.Identifier(str(resolved["hive"]))
                    ),
                    sql.SQL("{} AS source_timestamp").format(timestamp_expression),
                    sql.SQL("{} AS source_recorded_at").format(
                        sql.Identifier(str(resolved["timestamp"]))
                    ),
                    sql.SQL("{} AS internal_temperature").format(
                        sql.Identifier(str(resolved["temperature"]))
                    ),
                    sql.SQL("{} AS internal_humidity").format(
                        sql.Identifier(str(resolved["humidity"]))
                    ),
                    sql.SQL("{} AS internal_co2").format(
                        sql.Identifier(str(resolved["co2"]))
                    ),
                    sql.SQL("{} AS total_weight").format(
                        sql.Identifier(str(resolved["weight"]))
                    ),
                    self._optional_selection(
                        sql,
                        resolved["reading_at"],
                        "source_reading_at",
                        null_type="timestamptz",
                    ),
                    self._optional_selection(
                        sql,
                        resolved["external_temperature"],
                        "external_temperature",
                    ),
                    self._optional_selection(
                        sql,
                        resolved["external_humidity"],
                        "external_humidity",
                    ),
                    self._optional_selection(
                        sql,
                        resolved["battery_voltage"],
                        "battery_voltage",
                    ),
                ]

                query = sql.SQL(
                    "SELECT {selections} "
                    "FROM {schema}.{table} "
                    "WHERE {timestamp_expression} >= %s "
                    "AND {timestamp_expression} <= %s"
                ).format(
                    selections=sql.SQL(", ").join(selections),
                    schema=sql.Identifier(settings.schema),
                    table=sql.Identifier(settings.table),
                    timestamp_expression=timestamp_expression,
                )
                parameters: list[Any] = [start_utc, reference_timestamp]

                if selected_hive:
                    query += sql.SQL(" AND {} = %s").format(
                        sql.Identifier(str(resolved["hive"]))
                    )
                    parameters.append(selected_hive)

                query += sql.SQL(" ORDER BY {}, {}").format(
                    sql.Identifier(str(resolved["hive"])),
                    timestamp_expression,
                )

                with connection.cursor() as cursor:
                    cursor.execute(query, parameters)
                    rows = cursor.fetchall()
        except (LiveSensorConfigurationError, LiveSensorDatabaseError):
            raise
        except Exception as error:  # psycopg exposes driver-specific subclasses.
            raise LiveSensorDatabaseError(
                "Unable to read live beehive sensor rows from PostgreSQL. "
                "Check DATABASE_URL, SSL mode, table/column names, password "
                "URL encoding and network access."
            ) from error

        return pd.DataFrame(rows)

    def inspect_table(self, *, hive_id: str | None = None) -> dict[str, Any]:
        _, sql, _ = self._driver()
        selected_hive = hive_id or self.settings.configured_hive_id
        try:
            with self._connect() as connection:
                available = self._table_columns(connection)
                resolved = self._resolve_columns(available)
                timestamp_expression = self._timestamp_expression(sql, resolved)
                query = sql.SQL(
                    "SELECT COUNT(*) AS row_count, "
                    "COUNT(DISTINCT {hive}) AS hive_count, "
                    "MIN({timestamp}) AS earliest_timestamp, "
                    "MAX({timestamp}) AS latest_timestamp "
                    "FROM {schema}.{table}"
                ).format(
                    hive=sql.Identifier(str(resolved["hive"])),
                    timestamp=timestamp_expression,
                    schema=sql.Identifier(self.settings.schema),
                    table=sql.Identifier(self.settings.table),
                )
                parameters: list[Any] = []
                if selected_hive:
                    query += sql.SQL(" WHERE {} = %s").format(
                        sql.Identifier(str(resolved["hive"]))
                    )
                    parameters.append(selected_hive)
                with connection.cursor() as cursor:
                    cursor.execute(query, parameters)
                    summary = cursor.fetchone() or {}
        except (LiveSensorConfigurationError, LiveSensorDatabaseError):
            raise
        except Exception as error:
            raise LiveSensorDatabaseError(
                "Unable to inspect the configured PostgreSQL sensor table."
            ) from error

        optional_missing = {
            key: value
            for key, value in {
                "reading_at": self.settings.reading_at_column,
                "external_temperature": self.settings.external_temperature_column,
                "external_humidity": self.settings.external_humidity_column,
                "battery_voltage": self.settings.battery_voltage_column,
            }.items()
            if value and value not in available
        }
        latest = summary.get("latest_timestamp")
        latest_iso = pd.Timestamp(latest).isoformat() if latest is not None else None
        freshness_minutes = None
        if latest is not None:
            timestamp = pd.Timestamp(latest)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            freshness_minutes = max(
                0.0,
                (pd.Timestamp.now(tz="UTC") - timestamp).total_seconds() / 60.0,
            )

        return {
            "status": "ok",
            "schema": self.settings.schema,
            "table": self.settings.table,
            "configured_hive_id": selected_hive,
            "available_columns": sorted(available),
            "resolved_columns": resolved,
            "optional_configured_columns_not_found": optional_missing,
            "table_row_count": int(summary.get("row_count") or 0),
            "table_hive_count": int(summary.get("hive_count") or 0),
            "earliest_timestamp": (
                pd.Timestamp(summary["earliest_timestamp"]).isoformat()
                if summary.get("earliest_timestamp") is not None
                else None
            ),
            "latest_timestamp": latest_iso,
            "latest_freshness_minutes": freshness_minutes,
            "history_hours": self.settings.history_hours,
            "history_reference": self.settings.history_reference,
        }

    def connectivity_payload(self) -> dict[str, Any]:
        inspection = self.inspect_table()
        frame = self.fetch_recent()
        inspection.update(
            {
                "history_rows_returned": len(frame),
                "history_hives_returned": (
                    int(frame["source_hive_id"].astype(str).nunique())
                    if not frame.empty and "source_hive_id" in frame.columns
                    else 0
                ),
            }
        )
        return inspection
