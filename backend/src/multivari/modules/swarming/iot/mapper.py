from __future__ import annotations

from typing import Any


def to_model_reading(row: dict[str, Any]) -> dict[str, float]:
    """Map an IoT row to the eight sensor features used by the swarming model."""
    return {
        "internal_temperature_c": float(row.get("internal_temp") or 35.0),
        "internal_humidity_pct": float(row.get("internal_humidity") or 65.0),
        "co2_ppm": float(row.get("internal_co2") or 1200.0),
        "hive_weight_kg": float(row.get("total_weight") or 32.5),
        "external_temperature_c": float(row.get("external_temp") or 28.0),
        "external_humidity_pct": float(row.get("external_humidity") or 55.0),
        "rainfall_mm_hour": float(row.get("rainfall_mm_hour") or 0.0),
        "wind_speed_mps": float(row.get("wind_speed_mps") or 0.0),
    }
