from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .schema import HIVE_COLUMN, SENSOR_COLUMNS, TIMESTAMP_COLUMN


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    timestamp = pd.to_datetime(result[TIMESTAMP_COLUMN])
    result["hour"] = timestamp.dt.hour.astype("int8")
    result["day_of_week"] = timestamp.dt.dayofweek.astype("int8")
    result["month"] = timestamp.dt.month.astype("int8")
    result["day_of_year"] = timestamp.dt.dayofyear.astype("int16")
    result["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype("int8")
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24).astype("float32")
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24).astype("float32")
    result["day_of_year_sin"] = np.sin(2 * np.pi * result["day_of_year"] / 365.25).astype("float32")
    result["day_of_year_cos"] = np.cos(2 * np.pi * result["day_of_year"] / 365.25).astype("float32")
    return result


def build_common_features(
    df: pd.DataFrame,
    *,
    sensor_columns: Iterable[str] = SENSOR_COLUMNS,
    lags_hours: Iterable[int] = (1, 6, 24, 72),
    change_hours: Iterable[int] = (1, 6, 24, 72),
    rolling_windows_hours: Iterable[int] = (6, 24, 72),
    rolling_statistics: Iterable[str] = ("mean", "std"),
) -> pd.DataFrame:
    """Generate reusable past-only time-series features for all modules."""
    result = add_time_features(df)
    grouped = result.groupby(HIVE_COLUMN, sort=False)

    for sensor in sensor_columns:
        for lag in lags_hours:
            result[f"{sensor}_lag_{lag}h"] = grouped[sensor].shift(lag).astype("float32")

        for period in change_hours:
            result[f"{sensor}_change_{period}h"] = grouped[sensor].diff(period).astype("float32")

        for window in rolling_windows_hours:
            rolling = grouped[sensor].rolling(window=window, min_periods=window)
            if "mean" in rolling_statistics:
                result[f"{sensor}_roll_mean_{window}h"] = (
                    rolling.mean().reset_index(level=0, drop=True).astype("float32")
                )
            if "std" in rolling_statistics:
                result[f"{sensor}_roll_std_{window}h"] = (
                    rolling.std().reset_index(level=0, drop=True).astype("float32")
                )
            if "min" in rolling_statistics:
                result[f"{sensor}_roll_min_{window}h"] = (
                    rolling.min().reset_index(level=0, drop=True).astype("float32")
                )
            if "max" in rolling_statistics:
                result[f"{sensor}_roll_max_{window}h"] = (
                    rolling.max().reset_index(level=0, drop=True).astype("float32")
                )

    return result
