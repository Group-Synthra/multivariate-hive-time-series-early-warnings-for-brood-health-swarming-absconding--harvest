from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pandas as pd

from .config import IoTSettings


class IoTConfigurationError(RuntimeError):
    pass


class IoTRepositoryError(RuntimeError):
    pass


class PostgresIoTRepository:
    """Read-only Supabase/PostgreSQL repository for hive telemetry.

    Queries are relative to the latest timestamp stored for each device, rather
    than ``NOW()``. This keeps the integration working for both live streams and
    demonstration databases whose latest record may not equal the server clock.
    """

    REQUIRED_ALIASES = (
        "hive_id",
        "timestamp",
        "temperature_c",
        "humidity_pct",
        "co2_ppm",
        "weight_kg",
    )

    def __init__(self, settings: IoTSettings | None = None) -> None:
        self.settings = settings or IoTSettings.from_environment()

    @staticmethod
    def _driver():
        try:
            import psycopg
            from psycopg import sql
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise IoTConfigurationError(
                'psycopg is not installed. Run: python -m pip install -e ".[dev]"'
            ) from exc
        return psycopg, sql, dict_row

    def _require_url(self) -> None:
        if not self.settings.database_url:
            raise IoTConfigurationError(
                "DATABASE_URL is empty. Add the supplied Supabase PostgreSQL pooler URL to backend/.env."
            )

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        self._require_url()
        psycopg, _, dict_row = self._driver()
        kwargs: dict[str, Any] = {
            "connect_timeout": self.settings.connect_timeout_seconds,
            "row_factory": dict_row,
            "application_name": "hivorax-brood-health",
            # Supabase transaction pooler compatibility.
            "prepare_threshold": None,
        }
        if self.settings.sslmode:
            kwargs["sslmode"] = self.settings.sslmode
        try:
            with psycopg.connect(self.settings.database_url, **kwargs) as connection:
                connection.execute("SET TRANSACTION READ ONLY")
                yield connection
        except Exception as exc:
            raise IoTRepositoryError(f"Unable to read the IoT database: {exc}") from exc

    def _table_identifier(self, sql: Any) -> Any:
        return sql.Identifier(self.settings.schema, self.settings.table)

    def _contract(self) -> dict[str, str]:
        s = self.settings
        return {
            "hive_id": s.device_id_column,
            "timestamp": s.timestamp_column,
            "temperature_c": s.internal_temp_column,
            "humidity_pct": s.internal_humidity_column,
            "co2_ppm": s.internal_co2_column,
            "weight_kg": s.weight_column,
            "external_temp": s.external_temp_column,
            "external_humidity": s.external_humidity_column,
            "battery_voltage": s.battery_column,
            "reading_at": s.reading_at_column,
        }

    def _available_columns(self, connection: Any) -> set[str]:
        query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """
        with connection.cursor() as cursor:
            cursor.execute(query, (self.settings.schema, self.settings.table))
            return {row["column_name"] for row in cursor.fetchall()}

    def _validate(self, available: set[str]) -> None:
        contract = self._contract()
        missing = [contract[a] for a in self.REQUIRED_ALIASES if contract[a] not in available]
        if missing:
            raise IoTConfigurationError(
                f"IoT table {self.settings.schema}.{self.settings.table} is missing required columns: {missing}. "
                f"Available columns: {sorted(available)}"
            )

    def health(self) -> dict[str, Any]:
        _, sql, _ = self._driver()
        with self._connection() as connection:
            available = self._available_columns(connection)
            self._validate(available)
            table = self._table_identifier(sql)
            device = sql.Identifier(self.settings.device_id_column)
            timestamp = sql.Identifier(self.settings.timestamp_column)
            query = sql.SQL(
                "SELECT {device}::text AS device_id, {timestamp} AS recorded_at "
                "FROM {table} WHERE {timestamp} IS NOT NULL "
                "ORDER BY {timestamp} DESC LIMIT 1"
            ).format(device=device, timestamp=timestamp, table=table)
            with connection.cursor() as cursor:
                cursor.execute(query)
                latest = cursor.fetchone()
        return {
            "configured": True,
            "connected": True,
            "schema": self.settings.schema,
            "table": self.settings.table,
            "latest_device_id": str(latest["device_id"]) if latest else None,
            "latest_recorded_at": latest["recorded_at"].isoformat() if latest and latest["recorded_at"] else None,
            "available_columns": sorted(available),
            "mapped_columns": self._contract(),
            "refresh_seconds": self.settings.refresh_seconds,
            "feature_timezone": self.settings.feature_timezone,
        }

    def list_devices(self, *, lookback_hours: int | None = None) -> list[dict[str, Any]]:
        # Do not filter by NOW(): older but valid demonstration data must still appear.
        _, sql, _ = self._driver()
        with self._connection() as connection:
            available = self._available_columns(connection)
            self._validate(available)
            table = self._table_identifier(sql)
            device = sql.Identifier(self.settings.device_id_column)
            timestamp = sql.Identifier(self.settings.timestamp_column)
            query = sql.SQL(
                "SELECT {device}::text AS device_id, MAX({timestamp}) AS latest_recorded_at, "
                "COUNT(*) AS reading_count FROM {table} "
                "WHERE {device} IS NOT NULL AND {timestamp} IS NOT NULL "
                "GROUP BY {device} ORDER BY {device}::text"
            ).format(device=device, timestamp=timestamp, table=table)
            with connection.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        return [
            {
                "device_id": str(row["device_id"]),
                "hive": str(row["device_id"]),
                "latest_recorded_at": row["latest_recorded_at"].isoformat() if row["latest_recorded_at"] else None,
                "latest_timestamp": row["latest_recorded_at"].isoformat() if row["latest_recorded_at"] else None,
                "reading_count": int(row["reading_count"]),
            }
            for row in rows
        ]

    def fetch_history(self, device_id: str, *, lookback_hours: int | None = None) -> pd.DataFrame:
        _, sql, _ = self._driver()
        hours = max(24, min(int(lookback_hours or self.settings.lookback_hours), 24 * 30))
        device_id = str(device_id).strip()
        if not device_id or len(device_id) > 200:
            raise ValueError("A valid device_id is required")

        with self._connection() as connection:
            available = self._available_columns(connection)
            self._validate(available)
            contract = self._contract()
            selected = [(alias, source) for alias, source in contract.items() if source in available]
            columns = sql.SQL(", ").join(
                sql.SQL("{source} AS {alias}").format(
                    source=sql.Identifier(source), alias=sql.Identifier(alias)
                )
                for alias, source in selected
            )
            table = self._table_identifier(sql)
            device = sql.Identifier(self.settings.device_id_column)
            timestamp = sql.Identifier(self.settings.timestamp_column)
            query = sql.SQL(
                "WITH latest AS ("
                " SELECT MAX({timestamp}) AS latest_timestamp FROM {table} WHERE {device}::text = %s"
                ") SELECT {columns} FROM {table}, latest "
                "WHERE {device}::text = %s AND latest.latest_timestamp IS NOT NULL "
                "AND {timestamp} >= latest.latest_timestamp - (%s * INTERVAL '1 hour') "
                "AND {timestamp} IS NOT NULL ORDER BY {timestamp} ASC"
            ).format(timestamp=timestamp, table=table, device=device, columns=columns)
            with connection.cursor() as cursor:
                cursor.execute(query, (device_id, device_id, hours))
                rows = cursor.fetchall()
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["hive_id"] = frame["hive_id"].astype(str)
        return frame

    def fetch_between(
        self,
        device_id: str,
        *,
        start_timestamp: Any,
        end_timestamp: Any,
    ) -> pd.DataFrame:
        _, sql, _ = self._driver()
        device_id = str(device_id).strip()
        if not device_id:
            raise ValueError("A valid device_id is required")

        start = pd.Timestamp(start_timestamp)
        end = pd.Timestamp(end_timestamp)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

        with self._connection() as connection:
            available = self._available_columns(connection)
            self._validate(available)
            contract = self._contract()
            selected = [(alias, source) for alias, source in contract.items() if source in available]
            columns = sql.SQL(", ").join(
                sql.SQL("{source} AS {alias}").format(
                    source=sql.Identifier(source),
                    alias=sql.Identifier(alias),
                )
                for alias, source in selected
            )
            table = self._table_identifier(sql)
            device = sql.Identifier(self.settings.device_id_column)
            timestamp = sql.Identifier(self.settings.timestamp_column)
            query = sql.SQL(
                "SELECT {columns} FROM {table} "
                "WHERE {device}::text = %s "
                "AND {timestamp} >= %s AND {timestamp} <= %s "
                "AND {timestamp} IS NOT NULL "
                "ORDER BY {timestamp} ASC"
            ).format(
                columns=columns,
                table=table,
                device=device,
                timestamp=timestamp,
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (device_id, start.to_pydatetime(), end.to_pydatetime()),
                )
                rows = cursor.fetchall()

        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["hive_id"] = frame["hive_id"].astype(str)
        return frame

