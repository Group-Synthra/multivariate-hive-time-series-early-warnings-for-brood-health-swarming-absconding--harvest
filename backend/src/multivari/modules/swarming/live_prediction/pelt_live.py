"""Create hourly live PELT features for the swarming LSTM.

Raw IoT readings are resampled to one-hour averages.  The latest 24 hourly
rows are converted into the eight sensor plus four PELT features used by the
saved model, producing a DataFrame with shape ``(24, 12)``.
"""

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.preprocessing import StandardScaler

from .config import (
    FEATURE_COLUMNS,
    PELT_COLUMNS,
    PELT_MODEL,
    PELT_PEN,
    SEQUENCE_LENGTH,
)


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

# ``reading_at`` is the field used by the current IoT API.
TIMESTAMP_CANDIDATES = [
    "reading_at",
    "recorded_at",
    "timestamp",
    "created_at",
    "reading_time",
    "time",
    "datetime",
]


def _validate_readings(readings: list) -> None:
    """Validate the general structure of the received readings."""
    if not isinstance(readings, list):
        raise ValueError("readings must be a list of dictionaries.")
    if not readings:
        raise ValueError("No IoT readings were provided.")
    for index, reading in enumerate(readings):
        if not isinstance(reading, dict):
            raise ValueError(f"Reading at index {index} must be a dictionary.")


def _find_timestamp_column(df: pd.DataFrame) -> str:
    """Return the original name of the first recognized timestamp column."""
    normalized = {
        str(column).strip().lower(): column for column in df.columns
    }
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]

    raise ValueError(
        "No timestamp column was found in the IoT readings. "
        f"Received columns: {list(df.columns)}. "
        f"Expected one of: {TIMESTAMP_CANDIDATES}. "
        "Ensure routes.py includes reading_at when building readings_for_lstm."
    )


def _validate_sensor_columns(df: pd.DataFrame) -> None:
    """Ensure all model sensor columns are present."""
    missing = [column for column in SENSOR_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"IoT readings are missing sensor columns: {missing}")


def _fill_missing_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Interpolate time gaps, then forward/backward fill remaining values."""
    df = df.copy()
    df[columns] = df[columns].interpolate(
        method="time", limit_direction="both"
    )
    df[columns] = df[columns].ffill().bfill().fillna(0.0)
    return df


def _resample_to_hourly(readings: list) -> pd.DataFrame:
    """Convert raw IoT readings to the latest 24 hourly sensor averages."""
    df = pd.DataFrame(readings)
    timestamp_column = _find_timestamp_column(df)
    _validate_sensor_columns(df)

    if timestamp_column != "timestamp":
        df = df.rename(columns={timestamp_column: "timestamp"})

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        raise ValueError("No IoT reading contains a valid timestamp.")

    for column in SENSOR_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    hourly = (
        df.set_index("timestamp")[SENSOR_COLUMNS]
        .resample("1h")
        .mean()
    )
    hourly = _fill_missing_numeric(hourly, SENSOR_COLUMNS)
    hourly = hourly.tail(SEQUENCE_LENGTH)

    if len(hourly) < SEQUENCE_LENGTH:
        raise ValueError(
            f"At least {SEQUENCE_LENGTH} hours are required after resampling, "
            f"but only {len(hourly)} hourly records are available. For "
            "10-minute data, retrieve at least 144 readings covering 24 hours."
        )
    return hourly


def generate_pelt_features(readings: list) -> pd.DataFrame:
    """Return the latest 24 hourly rows with all 12 model features."""
    _validate_readings(readings)
    df = _resample_to_hourly(readings)

    missing_pelt = [column for column in PELT_COLUMNS if column not in df.columns]
    if missing_pelt:
        raise ValueError(f"PELT input columns are missing: {missing_pelt}")

    signal = df[PELT_COLUMNS].to_numpy(dtype=float)
    signal_scaled = StandardScaler().fit_transform(signal)
    n = len(signal_scaled)
    change_points = []

    if n >= 10:
        algorithm = rpt.Pelt(model=PELT_MODEL, min_size=2, jump=1)
        algorithm.fit(signal_scaled)
        change_points = [
            point for point in algorithm.predict(pen=PELT_PEN) if point < n
        ]

    breakpoints = np.zeros(n, dtype=float)
    for point in change_points:
        if 0 < point < n:
            breakpoints[point] = 1.0

    # Values below are hourly steps because the input has been resampled hourly.
    steps_since_breakpoint = []
    last_change = 0
    for index, value in enumerate(breakpoints):
        if value == 1.0:
            last_change = index
        steps_since_breakpoint.append(float(index - last_change))

    density = (
        pd.Series(breakpoints)
        .rolling(window=SEQUENCE_LENGTH, min_periods=1)
        .sum()
        .to_numpy()
    )

    duration = []
    counter = 0
    for value in breakpoints:
        if value == 1.0:
            counter = 0
        counter += 1
        duration.append(float(counter))

    df = df.copy()
    df["breakpoint"] = breakpoints
    # Retain this trained feature name; one unit now represents one hour.
    df["days_since_breakpoint"] = steps_since_breakpoint
    df["breakpoint_density"] = density
    df["segment_duration"] = duration

    result = df[FEATURE_COLUMNS].copy()
    expected = (SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    if result.shape != expected:
        raise ValueError(
            f"Expected feature shape {expected}, but received {result.shape}."
        )
    return result