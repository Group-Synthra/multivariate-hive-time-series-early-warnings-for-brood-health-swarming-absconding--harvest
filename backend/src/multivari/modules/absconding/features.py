from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from multivari.common.features import build_common_features
from multivari.common.schema import (
    HIVE_COLUMN,
    SENSOR_COLUMNS,
    TARGET_COLUMNS,
    TIMESTAMP_COLUMN,
)

from .config import AbscondingSettings

NON_FEATURE_COLUMNS = {
    TIMESTAMP_COLUMN,
    HIVE_COLUMN,
    *TARGET_COLUMNS,
    "split",
    "is_boundary_gap",
    "history_rows",
    "has_full_absconding_history",
}


def build_absconding_features(
    df: pd.DataFrame,
    settings: AbscondingSettings,
) -> pd.DataFrame:
    """Build current/past-only features for long-term colony deterioration."""
    ordered = df.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN]).copy()
    result = build_common_features(
        ordered,
        sensor_columns=SENSOR_COLUMNS,
        lags_hours=settings.lags_hours,
        change_hours=settings.change_hours,
        rolling_windows_hours=settings.rolling_windows_hours,
        rolling_statistics=settings.rolling_statistics,
    )

    grouped = result.groupby(HIVE_COLUMN, sort=False)
    additions: dict[str, pd.Series] = {}
    history_rows = grouped.cumcount().astype("int32") + 1
    additions["history_rows"] = history_rows
    additions["has_full_absconding_history"] = (
        history_rows >= settings.minimum_history_hours
    ).astype("int8")

    for sensor in SENSOR_COLUMNS:
        for window in settings.rolling_windows_hours:
            mean_column = f"{sensor}_roll_mean_{window}h"
            std_column = f"{sensor}_roll_std_{window}h"
            min_column = f"{sensor}_roll_min_{window}h"
            max_column = f"{sensor}_roll_max_{window}h"
            change_column = f"{sensor}_change_{window}h"

            if min_column in result and max_column in result:
                additions[f"{sensor}_roll_range_{window}h"] = (
                    result[max_column] - result[min_column]
                ).astype("float32")
            if mean_column in result and std_column in result:
                denominator = result[std_column].replace(0, np.nan)
                additions[f"{sensor}_z_{window}h"] = (
                    (result[sensor] - result[mean_column]) / denominator
                ).astype("float32")
            if change_column in result:
                additions[f"{sensor}_slope_{window}h"] = (
                    result[change_column] / float(window)
                ).astype("float32")

    for window in (24, 72, 168):
        weight_change = f"weight_kg_change_{window}h"
        weight_lag = f"weight_kg_lag_{window}h"
        if weight_change in result and weight_lag in result:
            denominator = result[weight_lag].abs().replace(0, np.nan)
            additions[f"weight_relative_change_{window}h"] = (
                result[weight_change] / denominator
            ).astype("float32")

    one_hour_weight_change = grouped["weight_kg"].diff()
    one_hour_co2_change = grouped["co2_ppm"].diff()
    for window in (24, 72, 168):
        weight_declining = one_hour_weight_change.lt(0).astype("float32")
        co2_rising = one_hour_co2_change.gt(0).astype("float32")
        additions[f"weight_decline_fraction_{window}h"] = (
            weight_declining.groupby(result[HIVE_COLUMN], sort=False)
            .rolling(window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )
        additions[f"co2_rise_fraction_{window}h"] = (
            co2_rising.groupby(result[HIVE_COLUMN], sort=False)
            .rolling(window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )

    instability_parts = []
    for sensor in ("temperature_c", "humidity_pct", "co2_ppm"):
        column = f"{sensor}_z_72h"
        if column in additions:
            instability_parts.append(additions[column].abs().clip(upper=5))
    weight_z = "weight_kg_z_72h"
    if weight_z in additions:
        instability_parts.append((-additions[weight_z]).clip(lower=0, upper=5))
    if instability_parts:
        additions["multisensor_instability_index"] = (
            pd.concat(instability_parts, axis=1).mean(axis=1) / 5.0
        ).astype("float32")

    additions["temperature_humidity_interaction"] = (
        result["temperature_c"] * result["humidity_pct"]
    ).astype("float32")

    # Report-aligned explainable stress indicators. These are operational
    # interpretation features, not universal biological cut-offs.
    temperature_stress = ((result["temperature_c"] - 35.0).abs() / 5.0).clip(0, 1)
    humidity_stress = ((result["humidity_pct"] - 60.0).abs() / 30.0).clip(0, 1)
    co2_stress = ((result["co2_ppm"] - 800.0) / 2200.0).clip(0, 1)
    weight_change_24h = result.get("weight_kg_change_24h", pd.Series(0.0, index=result.index))
    weight_change_72h = result.get("weight_kg_change_72h", pd.Series(0.0, index=result.index))
    weight_stress = pd.concat(
        [(-weight_change_24h / 2.0).clip(0, 1), (-weight_change_72h / 4.0).clip(0, 1)],
        axis=1,
    ).max(axis=1)
    environmental_stress = (
        0.28 * temperature_stress
        + 0.20 * humidity_stress
        + 0.24 * co2_stress
        + 0.28 * weight_stress
    ).clip(0, 1)
    additions["temperature_deviation_from_35"] = (result["temperature_c"] - 35.0).abs().astype("float32")
    additions["humidity_deviation_from_optimal"] = (result["humidity_pct"] - 60.0).abs().astype("float32")
    additions["co2_high_flag"] = result["co2_ppm"].ge(1800.0).astype("int8")
    additions["rapid_weight_loss_flag"] = weight_change_24h.le(-1.5).astype("int8")
    additions["sustained_weight_loss_24h"] = weight_change_24h.le(-0.5).astype("int8")
    additions["sustained_weight_loss_72h"] = weight_change_72h.le(-1.0).astype("int8")
    additions["environmental_stress_score"] = environmental_stress.astype("float32")
    additions["stress_trend_24h"] = (
        environmental_stress.groupby(result[HIVE_COLUMN], sort=False).diff(24).fillna(0.0)
    ).astype("float32")

    missing_sensor_count = result[list(SENSOR_COLUMNS)].isna().sum(axis=1).astype("int8")
    additions["missing_sensor_count"] = missing_sensor_count
    if "row_has_missing_sensor" not in result.columns:
        additions["row_has_missing_sensor"] = missing_sensor_count.gt(0).astype("int8")
    return pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1)


def select_feature_columns(
    df: pd.DataFrame,
    *,
    extra_excluded: Iterable[str] = (),
) -> list[str]:
    excluded = NON_FEATURE_COLUMNS | set(extra_excluded)
    columns = []
    for column in df.columns:
        if column in excluded or column.startswith("absconding_within_"):
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            columns.append(column)
    return columns
