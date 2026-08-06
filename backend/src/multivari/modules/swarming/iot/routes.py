from __future__ import annotations

import os

import pandas as pd
from flask import Blueprint, jsonify, request
from sqlalchemy import text

from .database import get_engine

iot_bp = Blueprint("swarming_iot", __name__)


def _identifier(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid database identifier: {name}")
    return name


@iot_bp.get("/api/iot/devices")
def devices():
    schema = _identifier(os.getenv("IOT_SCHEMA", "public"))
    table = _identifier(os.getenv("IOT_SENSOR_TABLE", "beehive_readings"))
    hive = _identifier(os.getenv("IOT_HIVE_COLUMN", "device_id"))
    query = text(f"SELECT DISTINCT {hive} FROM {schema}.{table} ORDER BY {hive}")
    frame = pd.read_sql(query, get_engine())
    return jsonify({"devices": frame[hive].dropna().tolist()})


@iot_bp.get("/api/iot/realtime-data")
def realtime():
    device_id = request.args.get("device_id")
    if not device_id:
        return jsonify({"error": "device_id is required"}), 400

    try:
        limit = min(max(int(request.args.get("limit", 432)), 1), 5000)
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    schema = _identifier(os.getenv("IOT_SCHEMA", "public"))
    table = _identifier(os.getenv("IOT_SENSOR_TABLE", "beehive_readings"))
    columns = {
        "hive": _identifier(os.getenv("IOT_HIVE_COLUMN", "device_id")),
        "time": _identifier(os.getenv("IOT_TIMESTAMP_COLUMN", "recorded_at")),
        "temperature": _identifier(os.getenv("IOT_TEMPERATURE_COLUMN", "internal_temp")),
        "humidity": _identifier(os.getenv("IOT_HUMIDITY_COLUMN", "internal_humidity")),
        "co2": _identifier(os.getenv("IOT_CO2_COLUMN", "internal_co2")),
        "weight": _identifier(os.getenv("IOT_WEIGHT_COLUMN", "total_weight")),
        "external_temperature": _identifier(
            os.getenv("IOT_EXTERNAL_TEMPERATURE_COLUMN", "external_temp")
        ),
        "external_humidity": _identifier(
            os.getenv("IOT_EXTERNAL_HUMIDITY_COLUMN", "external_humidity")
        ),
        "battery": _identifier(os.getenv("IOT_BATTERY_VOLTAGE_COLUMN", "battery_voltage")),
    }
    selected = ", ".join(columns[key] for key in columns if key != "hive")
    query = text(
        f"SELECT {selected} FROM {schema}.{table} "
        f"WHERE {columns['hive']} = :device ORDER BY {columns['time']} DESC LIMIT :limit"
    )
    frame = pd.read_sql(query, get_engine(), params={"device": device_id, "limit": limit})
    frame = frame.sort_values(columns["time"])
    return jsonify({"readings": frame.to_dict(orient="records")})
