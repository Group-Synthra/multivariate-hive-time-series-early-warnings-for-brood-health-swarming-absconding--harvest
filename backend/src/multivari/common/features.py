from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .schema import HIVE_COLUMN, SENSOR_COLUMNS, TIMESTAMP_COLUMN


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    timestamp = pd.to_datetime(result[TIMESTAMP_COLUMN])
    time_features = pd.DataFrame(index=result.index)
    time_features["hour"] = timestamp.dt.hour.astype("int8")
    time_features["day_of_week"] = timestamp.dt.dayofweek.astype("int8")
    time_features["month"] = timestamp.dt.month.astype("int8")
    time_features["day_of_year"] = timestamp.dt.dayofyear.astype("int16")
    time_features["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype("int8")
    time_features["hour_sin"] = np.sin(2 * np.pi * time_features["hour"] / 24).astype("float32")
    time_features["hour_cos"] = np.cos(2 * np.pi * time_features["hour"] / 24).astype("float32")
    time_features["day_of_year_sin"] = np.sin(
        2 * np.pi * time_features["day_of_year"] / 365.25
    ).astype("float32")
    time_features["day_of_year_cos"] = np.cos(
        2 * np.pi * time_features["day_of_year"] / 365.25
    ).astype("float32")
    return pd.concat([result, time_features], axis=1)


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
    additions: dict[str, pd.Series] = {}
    requested_statistics = set(rolling_statistics)

    for sensor in sensor_columns:
        for lag in lags_hours:
            additions[f"{sensor}_lag_{lag}h"] = grouped[sensor].shift(lag).astype("float32")

        for period in change_hours:
            additions[f"{sensor}_change_{period}h"] = grouped[sensor].diff(period).astype("float32")

        for window in rolling_windows_hours:
            rolling = grouped[sensor].rolling(window=window, min_periods=window)
            if "mean" in requested_statistics:
                additions[f"{sensor}_roll_mean_{window}h"] = (
                    rolling.mean().reset_index(level=0, drop=True).astype("float32")
                )
            if "std" in requested_statistics:
                additions[f"{sensor}_roll_std_{window}h"] = (
                    rolling.std().reset_index(level=0, drop=True).astype("float32")
                )
            if "min" in requested_statistics:
                additions[f"{sensor}_roll_min_{window}h"] = (
                    rolling.min().reset_index(level=0, drop=True).astype("float32")
                )
            if "max" in requested_statistics:
                additions[f"{sensor}_roll_max_{window}h"] = (
                    rolling.max().reset_index(level=0, drop=True).astype("float32")
                )

    return pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1)