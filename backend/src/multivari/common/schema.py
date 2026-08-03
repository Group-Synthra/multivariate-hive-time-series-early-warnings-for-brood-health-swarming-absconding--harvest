from __future__ import annotations

TIMESTAMP_COLUMN = "timestamp"
HIVE_COLUMN = "hive_id"

SENSOR_COLUMNS = (
    "temperature_c",
    "co2_ppm",
    "humidity_pct",
    "weight_kg",
)

TARGET_COLUMNS = (
    "brood_health_healthy_1",
    "swarming_happened_1",
    "absconding_happened_1",
    "honey_harvested_1",
)

REQUIRED_COLUMNS = (TIMESTAMP_COLUMN, HIVE_COLUMN, *SENSOR_COLUMNS, *TARGET_COLUMNS)

# Engineering sanity bounds, not biological health thresholds.
SENSOR_SANITY_BOUNDS = {
    "temperature_c": (-20.0, 60.0),
    "co2_ppm": (0.0, 100_000.0),
    "humidity_pct": (0.0, 100.0),
    "weight_kg": (0.0, 500.0),
}
