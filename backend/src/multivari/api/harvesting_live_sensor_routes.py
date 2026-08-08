from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
from flask import Flask, jsonify, request
from psycopg import sql
from psycopg.rows import dict_row


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _required_env(name: str) -> str:
    value = _env(name)
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    timestamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _domain_warnings() -> list[str]:
    backend_root = Path(__file__).resolve().parents[3]
    path = backend_root / "artifacts/reports/harvesting/live_iot_sensor_compatibility.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [str(item) for item in payload.get("warnings", []) if item]


def _latest_rows(hive_id: str | None) -> list[dict[str, Any]]:
    database_url = _required_env("DATABASE_URL")
    sslmode = _env("DATABASE_SSLMODE", "require")
    schema = _env("IOT_SCHEMA", "public")
    table = _env("IOT_SENSOR_TABLE", "beehive_readings")

    columns = {
        "hive_id": _env("IOT_HIVE_COLUMN", "device_id"),
        "timestamp": _env("IOT_TIMESTAMP_COLUMN", "recorded_at"),
        "internal_temperature_c": _env("IOT_TEMPERATURE_COLUMN", "internal_temp"),
        "internal_humidity_pct": _env("IOT_HUMIDITY_COLUMN", "internal_humidity"),
        "co2_ppm": _env("IOT_CO2_COLUMN", "internal_co2"),
        "weight_kg": _env("IOT_WEIGHT_COLUMN", "total_weight"),
        "external_temperature_c": _env("IOT_EXTERNAL_TEMPERATURE_COLUMN", "external_temp"),
        "external_humidity_pct": _env("IOT_EXTERNAL_HUMIDITY_COLUMN", "external_humidity"),
        "battery_voltage": _env("IOT_BATTERY_VOLTAGE_COLUMN", "battery_voltage"),
    }

    select_items = [
        sql.SQL("{} AS {}").format(
            sql.Identifier(database_name),
            sql.Identifier(output_name),
        )
        for output_name, database_name in columns.items()
    ]

    where_clause = sql.SQL("")
    params: list[Any] = []
    if hive_id:
        where_clause = sql.SQL("WHERE {} = %s").format(sql.Identifier(columns["hive_id"]))
        params.append(hive_id)

    query = sql.SQL(
        """
        WITH ranked AS (
            SELECT
                {select_items},
                ROW_NUMBER() OVER (
                    PARTITION BY {hive_column}
                    ORDER BY {timestamp_column} DESC
                ) AS row_number
            FROM {schema}.{table}
            {where_clause}
        )
        SELECT {output_columns}
        FROM ranked
        WHERE row_number = 1
        ORDER BY hive_id
        """
    ).format(
        select_items=sql.SQL(", ").join(select_items),
        hive_column=sql.Identifier(columns["hive_id"]),
        timestamp_column=sql.Identifier(columns["timestamp"]),
        schema=sql.Identifier(schema),
        table=sql.Identifier(table),
        where_clause=where_clause,
        output_columns=sql.SQL(", ").join(sql.Identifier(name) for name in columns),
    )

    with (
        psycopg.connect(
            database_url,
            sslmode=sslmode,
            row_factory=dict_row,
        ) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(query, params)
        return list(cursor.fetchall())


def _serialize(
    row: dict[str, Any],
    *,
    interval_minutes: int,
    stale_after_minutes: int,
) -> dict[str, Any]:
    timestamp = _as_utc(row.get("timestamp"))
    freshness_minutes = None
    if timestamp is not None:
        freshness_minutes = max(
            0.0,
            (datetime.now(UTC) - timestamp).total_seconds() / 60.0,
        )

    return {
        "hive_id": str(row.get("hive_id", "")),
        "timestamp": timestamp.isoformat() if timestamp else None,
        "next_expected_at": (
            (timestamp + timedelta(minutes=interval_minutes)).isoformat() if timestamp else None
        ),
        "freshness_minutes": freshness_minutes,
        "freshness_label": (
            "Fresh"
            if freshness_minutes is not None and freshness_minutes <= stale_after_minutes
            else "Stale"
        ),
        "internal_temperature_c": _as_float(row.get("internal_temperature_c")),
        "internal_humidity_pct": _as_float(row.get("internal_humidity_pct")),
        "co2_ppm": _as_float(row.get("co2_ppm")),
        "weight_kg": _as_float(row.get("weight_kg")),
        "external_temperature_c": _as_float(row.get("external_temperature_c")),
        "external_humidity_pct": _as_float(row.get("external_humidity_pct")),
        "battery_voltage": _as_float(row.get("battery_voltage")),
    }


def register_harvesting_live_sensor_routes(app: Flask) -> None:
    endpoint = "harvesting_live_sensor_snapshot"
    if endpoint in app.view_functions:
        return

    @app.get("/api/harvesting/live-sensors", endpoint=endpoint)
    def harvesting_live_sensor_snapshot():
        hive_id = request.args.get("hive_id", "").strip() or None
        interval_minutes = int(_env("IOT_INTERVAL_MINUTES", "10"))
        stale_after_minutes = int(_env("IOT_STALE_AFTER_MINUTES", "30"))

        try:
            snapshots = [
                _serialize(
                    row,
                    interval_minutes=interval_minutes,
                    stale_after_minutes=stale_after_minutes,
                )
                for row in _latest_rows(hive_id)
            ]
        except Exception as error:
            app.logger.exception("Could not load live sensor snapshots.")
            return (
                jsonify(
                    {
                        "status": "live_sensor_snapshot_error",
                        "message": str(error),
                        "latest_sensor_by_hive": [],
                    }
                ),
                500,
            )

        return jsonify(
            {
                "status": "live_sensor_snapshot_ready",
                "generated_at": datetime.now(UTC).isoformat(),
                "source": "PostgreSQL IoT",
                "available_hives": [item["hive_id"] for item in snapshots],
                "latest_sensor_by_hive": snapshots,
                "domain_warnings": _domain_warnings(),
            }
        )
