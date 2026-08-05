from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .scoring import compute_score_components, health_level_code

FEATURE_SCHEMA_VERSION = "brood-score-v3.0"
TARGET_COLUMN = "brood_health_healthy_1"
SENSORS = ("temperature_c", "humidity_pct", "co2_ppm", "weight_kg")
ENVIRONMENT_SENSORS = ("temperature_c", "humidity_pct", "co2_ppm")
LAGS = (1, 6, 12, 24, 48, 72)
ROLLING_WINDOWS = (6, 12, 24, 72)
CHANGES = (1, 6, 12, 24)
MINIMUM_TRAINING_HISTORY_HOURS = 72

PHYSICAL_RANGES = {
    "temperature_c": (-10.0, 60.0),
    "humidity_pct": (0.0, 100.0),
    "co2_ppm": (0.0, 100_000.0),
    "weight_kg": (0.1, 300.0),
    "external_temp": (-20.0, 65.0),
    "external_humidity": (0.0, 100.0),
    "battery_voltage": (0.0, 30.0),
}

CANONICAL_IOT_MAPPING = {
    "device_id": "hive_id",
    "recorded_at": "timestamp",
    "internal_temp": "temperature_c",
    "internal_humidity": "humidity_pct",
    "internal_co2": "co2_ppm",
    "total_weight": "weight_kg",
    "external_temp": "external_temp",
    "external_humidity": "external_humidity",
    "battery_voltage": "battery_voltage",
    "reading_at": "reading_at",
}


def _coerce_and_validate_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column, (minimum, maximum) in PHYSICAL_RANGES.items():
        if column not in out.columns:
            continue
        out[column] = pd.to_numeric(out[column], errors="coerce")
        invalid = ~out[column].between(minimum, maximum) & out[column].notna()
        if invalid.any():
            out.loc[invalid, column] = np.nan
    return out


def normalise_historical(frame: pd.DataFrame) -> pd.DataFrame:
    """Module-specific preprocessing applied after the common clean dataset.

    It keeps the common pipeline as the shared first phase, then adds brood-specific
    range validation, causal short-gap filling and consistent hourly ordering. Only
    forward filling is used so later readings never leak into earlier features.
    """

    required = {"hive_id", "timestamp", *SENSORS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Historical brood-health data are missing columns: {missing}")

    out = frame.copy()
    out["hive_id"] = out["hive_id"].astype("string").str.strip()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = _coerce_and_validate_ranges(out)
    if TARGET_COLUMN in out.columns:
        out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
        out.loc[~out[TARGET_COLUMN].isin([0, 1]), TARGET_COLUMN] = np.nan

    out = (
        out.dropna(subset=["hive_id", "timestamp"])
        .sort_values(["hive_id", "timestamp"])
        .drop_duplicates(["hive_id", "timestamp"], keep="last")
        .reset_index(drop=True)
    )

    for sensor in SENSORS:
        out[f"{sensor}_was_missing"] = out[sensor].isna().astype(int)
    out[list(SENSORS)] = out.groupby("hive_id", sort=False)[list(SENSORS)].ffill(limit=2)
    return out


def map_iot_frame(
    raw: pd.DataFrame,
    *,
    column_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    mapping = dict(column_mapping or CANONICAL_IOT_MAPPING)
    available_mapping = {source: target for source, target in mapping.items() if source in raw.columns}
    out = raw.rename(columns=available_mapping).copy()
    required = {"hive_id", "timestamp", *SENSORS}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"Live IoT data are missing required mapped columns: {missing}")
    return normalise_historical(out)


def aggregate_live_hourly(frame: pd.DataFrame, *, frequency: str = "1h") -> pd.DataFrame:
    """Aggregate approximately 10-minute IoT readings to the hourly training schema."""

    if frame.empty:
        return frame.copy()
    numeric = [
        column
        for column in (*SENSORS, "external_temp", "external_humidity", "battery_voltage")
        if column in frame.columns
    ]
    pieces: list[pd.DataFrame] = []
    for hive_id, group in frame.groupby("hive_id", sort=False):
        indexed = group.set_index("timestamp").sort_index()
        hourly = indexed[numeric].resample(frequency).median()
        hourly["raw_reading_count"] = indexed[numeric[0]].resample(frequency).count()
        hourly["hive_id"] = hive_id
        pieces.append(hourly.reset_index())
    result = pd.concat(pieces, ignore_index=True)
    return normalise_historical(result)


def _rolling_feature(series: pd.Series, groups: pd.Series, window: int, statistic: str) -> pd.Series:
    # Shift first: every rolling statistic contains only observations strictly before t.
    shifted = series.groupby(groups, sort=False).shift(1)
    grouped = shifted.groupby(groups, sort=False)
    min_periods = max(3, window // 3)
    if statistic == "mean":
        return grouped.transform(lambda values: values.rolling(window, min_periods=min_periods).mean())
    if statistic == "std":
        return grouped.transform(lambda values: values.rolling(window, min_periods=min_periods).std(ddof=0))
    if statistic == "minimum":
        return grouped.transform(lambda values: values.rolling(window, min_periods=min_periods).min())
    if statistic == "maximum":
        return grouped.transform(lambda values: values.rolling(window, min_periods=min_periods).max())
    if statistic == "median":
        return grouped.transform(lambda values: values.rolling(window, min_periods=min_periods).median())
    raise ValueError(f"Unsupported rolling statistic: {statistic}")


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal brood-health features without target, hive ID or absolute date."""

    data = normalise_historical(frame)
    hive = data["hive_id"]
    features = pd.DataFrame(index=data.index)

    for sensor in ENVIRONMENT_SENSORS:
        values = pd.to_numeric(data[sensor], errors="coerce")
        features[sensor] = values
        features[f"{sensor}_missing"] = data.get(f"{sensor}_was_missing", values.isna()).astype(float)
        grouped = values.groupby(hive, sort=False)
        for lag in LAGS:
            features[f"{sensor}_lag_{lag}h"] = grouped.shift(lag)
        for window in ROLLING_WINDOWS:
            features[f"{sensor}_mean_{window}h"] = _rolling_feature(values, hive, window, "mean")
            features[f"{sensor}_std_{window}h"] = _rolling_feature(values, hive, window, "std")
            if window in (24, 72):
                features[f"{sensor}_min_{window}h"] = _rolling_feature(values, hive, window, "minimum")
                features[f"{sensor}_max_{window}h"] = _rolling_feature(values, hive, window, "maximum")
        for change in CHANGES:
            features[f"{sensor}_change_{change}h"] = values - grouped.shift(change)

    # Absolute hive weight can identify a colony and transfer poorly to unseen hives.
    # Brood-specific weight features therefore represent relative change and stability.
    weight = pd.to_numeric(data["weight_kg"], errors="coerce")
    weight_group = weight.groupby(hive, sort=False)
    features["weight_missing"] = data.get("weight_kg_was_missing", weight.isna()).astype(float)
    for lag in (1, 6, 24, 72):
        previous = weight_group.shift(lag)
        features[f"weight_change_{lag}h"] = weight - previous
        features[f"weight_change_pct_{lag}h"] = (weight - previous) / previous.abs().clip(lower=1.0)
    for window in ROLLING_WINDOWS:
        median = _rolling_feature(weight, hive, window, "median")
        std = _rolling_feature(weight, hive, window, "std")
        features[f"weight_relative_to_median_{window}h"] = (weight - median) / median.abs().clip(lower=1.0)
        features[f"weight_cv_{window}h"] = std / median.abs().clip(lower=1.0)

    hour = data["timestamp"].dt.hour.astype(float)
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    features["is_night"] = ((hour < 6) | (hour >= 18)).astype(int)

    features["temperature_deviation_35"] = (data["temperature_c"] - 35.0).abs()
    features["humidity_deviation_65"] = (data["humidity_pct"] - 65.0).abs()
    features["co2_log1p"] = np.log1p(data["co2_ppm"].clip(lower=0.0))
    features["temperature_humidity_interaction"] = data["temperature_c"] * data["humidity_pct"]
    features["temperature_co2_interaction"] = data["temperature_c"] * features["co2_log1p"]
    features["history_hours"] = data.groupby("hive_id", sort=False).cumcount().astype(float)

    # Deliberately absent: brood_health_healthy_1, hive_id, full date/day-of-year and
    # future readings. This prevents direct label leakage and date/hive memorisation.
    return features.replace([np.inf, -np.inf], np.nan)


def _future_window_minimum(series: pd.Series, groups: pd.Series, horizon_hours: int) -> pd.Series:
    shifted = [series.groupby(groups, sort=False).shift(-offset) for offset in range(1, horizon_hours + 1)]
    return pd.concat(shifted, axis=1).min(axis=1, skipna=False)


def build_supervised_dataset(
    frame: pd.DataFrame,
    *,
    horizon_hours: int = 6,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Create the early-warning regression task.

    Target: the *minimum* Brood Health Score observed in the next forecast window.
    This avoids the trivial question "will the stable binary label remain unchanged?"
    and focuses the model on upcoming deterioration.
    """

    horizon = int(horizon_hours)
    if not 1 <= horizon <= 168:
        raise ValueError("horizon_hours must be between 1 and 168")
    data = normalise_historical(frame)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Training data do not contain {TARGET_COLUMN}")

    scored = compute_score_components(data)
    features = build_feature_frame(data)
    future_score = _future_window_minimum(scored["brood_health_score"], data["hive_id"], horizon)
    target_timestamp = data["timestamp"].groupby(data["hive_id"], sort=False).shift(-horizon)

    future_binary = _future_window_minimum(
        pd.to_numeric(data[TARGET_COLUMN], errors="coerce"), data["hive_id"], horizon
    )
    current_score = scored["brood_health_score"]
    current_level = health_level_code(current_score)
    future_level = health_level_code(future_score.fillna(current_score))
    score_drop = current_score - future_score

    metadata = data[["hive_id", "timestamp"]].copy()
    metadata["target_timestamp"] = target_timestamp
    metadata["current_score"] = current_score
    metadata["current_level_code"] = current_level
    metadata["future_level_code"] = future_level
    metadata["future_observed_healthy"] = future_binary
    metadata["score_drop"] = score_drop
    metadata["transition_window"] = (
        (metadata["current_level_code"] != metadata["future_level_code"])
        | (metadata["score_drop"] >= 10.0)
    )
    metadata["history_hours"] = data.groupby("hive_id", sort=False).cumcount()

    keep = (
        future_score.notna()
        & metadata["target_timestamp"].notna()
        & metadata["future_observed_healthy"].notna()
        & (metadata["history_hours"] >= MINIMUM_TRAINING_HISTORY_HOURS)
    )
    x = features.loc[keep].reset_index(drop=True)
    y = future_score.loc[keep].astype(float).reset_index(drop=True)
    meta = metadata.loc[keep].reset_index(drop=True)
    columns = list(x.columns)
    forbidden = {TARGET_COLUMN, "hive_id", "timestamp", "target_timestamp"}
    leaked = forbidden.intersection(columns)
    if leaked:
        raise RuntimeError(f"Forbidden columns entered the brood-health feature matrix: {sorted(leaked)}")
    return x, y, meta, columns


def build_latest_inference_row(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = normalise_historical(frame)
    features = build_feature_frame(data)
    latest_positions = data.groupby("hive_id", sort=False)["timestamp"].idxmax()
    latest_meta = data.loc[latest_positions, ["hive_id", "timestamp", *SENSORS]].copy()
    latest_features = features.loc[latest_positions].reindex(columns=feature_columns)
    latest_features.index = latest_meta.index
    return latest_features.reset_index(drop=True), latest_meta.reset_index(drop=True)
