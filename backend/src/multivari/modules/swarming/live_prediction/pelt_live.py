"""
=========================================================
Live Prediction — PELT Feature Generator
=========================================================

Responsibility:
  Given the latest 24 sensor readings, reproduce the exact
  PELT feature engineering applied during training:
    - breakpoint
    - days_since_breakpoint
    - breakpoint_density (rolling-24 window)
    - segment_duration

Input:
  readings : list of 24 dicts, each containing the 8 sensor keys

Output:
  pd.DataFrame with all 12 feature columns (sensor + PELT),
  in the exact order defined by FEATURE_COLUMNS.
=========================================================
"""

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.preprocessing import StandardScaler

from .config import (
    PELT_COLUMNS,
    PELT_MODEL,
    PELT_PEN,
    SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
)


# -------------------------------------------------------
# Sensor columns (the 8 raw sensor features)
# -------------------------------------------------------
SENSOR_COLUMNS = [
    "internal_temperature_c",
    "internal_humidity_pct",
    "co2_ppm",
    "hive_weight_kg",
    "external_temperature_c",
    "external_humidity_pct",
    "rainfall_mm_hour",
    "wind_speed_mps",
]


def _validate_readings(readings: list) -> None:
    """Raise ValueError with a clear message if readings are invalid."""
    if not isinstance(readings, list):
        raise ValueError("readings must be a list of dicts.")

    if len(readings) < SEQUENCE_LENGTH:
        raise ValueError(
            f"At least {SEQUENCE_LENGTH} readings required. "
            f"Got {len(readings)}."
        )

    required_keys = set(SENSOR_COLUMNS)
    for i, row in enumerate(readings[-SEQUENCE_LENGTH:]):
        missing = required_keys - set(row.keys())
        if missing:
            raise ValueError(
                f"Reading at index {i} is missing keys: {missing}"
            )


def _fill_missing_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Forward-fill → backward-fill → column mean for NaN values."""
    df[columns] = df[columns].ffill()
    df[columns] = df[columns].bfill()
    # If still NaN (all-NaN column), fill with 0
    df[columns] = df[columns].fillna(0.0)
    return df


def generate_pelt_features(readings: list) -> pd.DataFrame:
    """
    Generate the four PELT features for the latest 24 readings.

    Parameters
    ----------
    readings : list of dict
        Must contain at least SEQUENCE_LENGTH entries.
        Each dict must have the 8 sensor keys.

    Returns
    -------
    pd.DataFrame
        Shape (24, 12) — 8 sensor columns + 4 PELT columns,
        in FEATURE_COLUMNS order.
    """
    _validate_readings(readings)

    # Use only the last 24 readings
    window = readings[-SEQUENCE_LENGTH:]

    df = pd.DataFrame(window)

    # Ensure all sensor columns exist and are numeric
    for col in SENSOR_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _fill_missing_numeric(df, SENSOR_COLUMNS)

    # -------------------------------------------------------
    # Prepare PELT signal  (same as training)
    # -------------------------------------------------------
    signal = df[PELT_COLUMNS].values.astype(float)

    # Scale PELT signal in-place (training scaled per-hive)
    local_scaler = StandardScaler()
    signal_scaled = local_scaler.fit_transform(signal)

    # -------------------------------------------------------
    # Run PELT
    # -------------------------------------------------------
    n = len(signal_scaled)
    change_points = []

    if n >= 10:   # ruptures needs a minimum length
        algorithm = rpt.Pelt(model=PELT_MODEL, min_size=2, jump=1)
        algorithm.fit(signal_scaled)
        raw_cp = algorithm.predict(pen=PELT_PEN)
        # ruptures always appends n as the last element — remove it
        change_points = [cp for cp in raw_cp if cp < n]

    # -------------------------------------------------------
    # Build breakpoint binary array
    # -------------------------------------------------------
    breakpoint_array = np.zeros(n, dtype=float)
    for cp in change_points:
        if 0 < cp < n:
            breakpoint_array[cp] = 1.0

    # -------------------------------------------------------
    # days_since_breakpoint
    # -------------------------------------------------------
    days_since = []
    last_change = 0
    for i, val in enumerate(breakpoint_array):
        if val == 1.0:
            last_change = i
        days_since.append(float(i - last_change))

    # -------------------------------------------------------
    # breakpoint_density  (rolling window of 24)
    # -------------------------------------------------------
    density = (
        pd.Series(breakpoint_array)
        .rolling(window=SEQUENCE_LENGTH, min_periods=1)
        .sum()
        .values
    )

    # -------------------------------------------------------
    # segment_duration
    # -------------------------------------------------------
    duration = []
    counter = 0
    for val in breakpoint_array:
        if val == 1.0:
            counter = 0
        counter += 1
        duration.append(float(counter))

    # -------------------------------------------------------
    # Assemble result DataFrame
    # -------------------------------------------------------
    df["breakpoint"]           = breakpoint_array
    df["days_since_breakpoint"] = days_since
    df["breakpoint_density"]   = density
    df["segment_duration"]     = duration

    # Return only the 12 model features in the correct order
    result = df[FEATURE_COLUMNS].copy()
    return result
