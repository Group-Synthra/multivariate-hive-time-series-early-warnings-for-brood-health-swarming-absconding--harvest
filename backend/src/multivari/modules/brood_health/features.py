from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .scoring import BroodHealthScoreConfig, compute_score_components, health_level_code

FEATURE_SCHEMA_VERSION = "brood-score-v6.0"
TARGET_COLUMN = "brood_health_healthy_1"
SENSORS = ("temperature_c", "humidity_pct", "co2_ppm", "weight_kg")
ENVIRONMENT_SENSORS = ("temperature_c", "humidity_pct", "co2_ppm")
LAGS = (1, 3, 6, 12, 24, 48, 72)
ROLLING_WINDOWS = (3, 6, 12, 24, 72)
CHANGES = (1, 3, 6, 12, 24)
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
        out.loc[invalid, column] = np.nan
    return out


def normalise_historical(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply brood-specific preprocessing after the shared common pipeline.

    The common clean table remains unchanged. This function applies only brood-health
    range validation, timestamp standardisation and causal short-gap forward filling.
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
        out[f"{sensor}_was_missing"] = out[sensor].isna().astype("int8")
    # Forward-only, maximum two hourly gaps. No future interpolation is used.
    out[list(SENSORS)] = out.groupby("hive_id", sort=False)[list(SENSORS)].ffill(limit=2)
    return out


def map_iot_frame(
    raw: pd.DataFrame,
    *,
    column_mapping: Mapping[str, str] | None = None,
    weight_scale_factor: float = 1.0,
    weight_offset_kg: float = 0.0,
) -> pd.DataFrame:
    mapping = dict(column_mapping or CANONICAL_IOT_MAPPING)
    available_mapping = {
        source: target for source, target in mapping.items() if source in raw.columns
    }
    out = raw.rename(columns=available_mapping).copy()
    required = {"hive_id", "timestamp", *SENSORS}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"Live IoT data are missing required mapped columns: {missing}")

    out["weight_kg"] = pd.to_numeric(out["weight_kg"], errors="coerce") * float(
        weight_scale_factor
    ) + float(weight_offset_kg)
    return normalise_historical(out)


def aggregate_live_hourly(
    frame: pd.DataFrame,
    *,
    frequency: str = "1h",
    anchor_to_latest: bool = True,
) -> pd.DataFrame:
    """Aggregate live readings to complete rolling one-hour windows.

    Historical training observations are hourly. During deployment the latest raw IoT
    timestamp may be 10:32, 10:42, and so on. When ``anchor_to_latest`` is enabled,
    complete one-hour windows are aligned to that latest timestamp:

        (09:32, 10:32] -> 10:32
        (08:32, 09:32] -> 09:32

    This preserves one-hour spacing while allowing the six-hour forecast target to move
    with each new reading. No synthetic ten-minute training rows are created.
    """

    if frame.empty:
        return frame.copy()
    if str(frequency).lower() not in {"1h", "60min", "60t"}:
        raise ValueError("Live brood-health aggregation currently requires one-hour windows")

    numeric = [
        column
        for column in (*SENSORS, "external_temp", "external_humidity", "battery_voltage")
        if column in frame.columns
    ]
    pieces: list[pd.DataFrame] = []
    for hive_id, group in frame.groupby("hive_id", sort=False):
        ordered = group.sort_values("timestamp").copy()
        ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce", utc=True)
        ordered = ordered.dropna(subset=["timestamp"])
        if ordered.empty:
            continue

        if anchor_to_latest:
            anchor = pd.Timestamp(ordered["timestamp"].max())
            seconds_back = (anchor - ordered["timestamp"]).dt.total_seconds()
            bucket_index = np.floor(
                np.maximum(seconds_back.to_numpy(dtype=float), 0.0) / 3600.0
            ).astype(int)
            ordered["_bucket_end"] = anchor - pd.to_timedelta(bucket_index, unit="h")
            aggregated = (
                ordered.groupby("_bucket_end", sort=True, observed=True)[numeric]
                .median()
                .reset_index()
                .rename(columns={"_bucket_end": "timestamp"})
            )
            counts = (
                ordered.groupby("_bucket_end", sort=True, observed=True)[numeric[0]]
                .count()
                .rename("raw_reading_count")
                .reset_index()
                .rename(columns={"_bucket_end": "timestamp"})
            )
            hourly = aggregated.merge(counts, on="timestamp", how="left")
        else:
            indexed = ordered.set_index("timestamp")
            hourly = indexed[numeric].resample(frequency, label="right", closed="right").median()
            hourly["raw_reading_count"] = (
                indexed[numeric[0]].resample(frequency, label="right", closed="right").count()
            )
            hourly = hourly.reset_index()

        hourly["hive_id"] = hive_id
        pieces.append(hourly)

    if not pieces:
        return frame.iloc[0:0].copy()

    result = pd.concat(pieces, ignore_index=True)

    normalised = normalise_historical(result)
    normalised.attrs["aggregation_strategy"] = (
        "rolling_one_hour_median_aligned_to_latest_reading"
        if anchor_to_latest
        else "clock_hour_median"
    )
    return normalised


def _rolling_from_shifted(
    shifted: pd.Series,
    group_codes: pd.Series,
    window: int,
    statistic: str,
) -> pd.Series:
    """Efficient grouped rolling statistic for already shifted values."""

    min_periods = max(3, window // 3)
    rolling = shifted.groupby(group_codes, sort=False).rolling(
        window,
        min_periods=min_periods,
    )
    if statistic == "mean":
        result = rolling.mean()
    elif statistic == "std":
        result = rolling.std(ddof=0)
    elif statistic == "minimum":
        result = rolling.min()
    elif statistic == "maximum":
        result = rolling.max()
    elif statistic == "median":
        result = rolling.median()
    else:
        raise ValueError(f"Unsupported rolling statistic: {statistic}")
    return result.reset_index(level=0, drop=True).reindex(shifted.index)


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal features without labels, IDs, absolute date or future values.

    Columns are accumulated in a dictionary and concatenated once. This avoids
    DataFrame fragmentation during full-dataset feature generation.
    """

    data = normalise_historical(frame)
    hive = data["hive_id"]
    group_codes = pd.Series(pd.factorize(hive, sort=False)[0], index=data.index)
    columns: dict[str, pd.Series | np.ndarray] = {}

    for sensor in ENVIRONMENT_SENSORS:
        values = pd.to_numeric(data[sensor], errors="coerce")
        columns[sensor] = values
        columns[f"{sensor}_missing"] = data.get(f"{sensor}_was_missing", values.isna()).astype(
            float
        )
        grouped = values.groupby(group_codes, sort=False)
        shifted = grouped.shift(1)

        for lag in LAGS:
            columns[f"{sensor}_lag_{lag}h"] = grouped.shift(lag)
        for window in ROLLING_WINDOWS:
            columns[f"{sensor}_mean_{window}h"] = _rolling_from_shifted(
                shifted, group_codes, window, "mean"
            )
            columns[f"{sensor}_std_{window}h"] = _rolling_from_shifted(
                shifted, group_codes, window, "std"
            )
            if window in (24, 72):
                columns[f"{sensor}_min_{window}h"] = _rolling_from_shifted(
                    shifted, group_codes, window, "minimum"
                )
                columns[f"{sensor}_max_{window}h"] = _rolling_from_shifted(
                    shifted, group_codes, window, "maximum"
                )
        for change in CHANGES:
            columns[f"{sensor}_change_{change}h"] = values - grouped.shift(change)

    # Absolute weight is deliberately excluded. Only scale-invariant movement and
    # stability are used so different hive construction and tare weight transfer better.
    weight = pd.to_numeric(data["weight_kg"], errors="coerce")
    weight_group = weight.groupby(group_codes, sort=False)
    weight_shifted = weight_group.shift(1)
    columns["weight_missing"] = data.get("weight_kg_was_missing", weight.isna()).astype(float)
    for lag in (1, 3, 6, 12, 24, 48, 72):
        previous = weight_group.shift(lag)
        columns[f"weight_change_pct_{lag}h"] = (weight - previous) / previous.abs().clip(lower=1.0)
    for window in ROLLING_WINDOWS:
        median = _rolling_from_shifted(weight_shifted, group_codes, window, "median")
        std = _rolling_from_shifted(weight_shifted, group_codes, window, "std")
        columns[f"weight_relative_to_median_{window}h"] = (weight - median) / median.abs().clip(
            lower=1.0
        )
        columns[f"weight_cv_{window}h"] = std / median.abs().clip(lower=1.0)

    hour = data["timestamp"].dt.hour.astype(float)
    columns["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    columns["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    columns["is_night"] = ((hour < 6) | (hour >= 18)).astype("int8")

    columns["temperature_deviation_35"] = (data["temperature_c"] - 35.0).abs()
    columns["humidity_deviation_65"] = (data["humidity_pct"] - 65.0).abs()
    columns["co2_log1p"] = np.log1p(data["co2_ppm"].clip(lower=0.0))
    columns["temperature_humidity_interaction"] = data["temperature_c"] * data["humidity_pct"]
    columns["temperature_co2_interaction"] = data["temperature_c"] * columns["co2_log1p"]
    columns["history_hours"] = data.groupby("hive_id", sort=False).cumcount().astype(float)

    features = pd.DataFrame(columns, index=data.index)
    return features.replace([np.inf, -np.inf], np.nan)


def target_columns(horizon_hours: int) -> list[str]:
    return [f"score_t_plus_{hour}h" for hour in range(1, int(horizon_hours) + 1)]


def build_supervised_dataset(
    frame: pd.DataFrame,
    *,
    horizon_hours: int = 6,
    score_config: BroodHealthScoreConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Create a multi-horizon regression task.

    Primary research output: exact Brood Health Score at t + horizon.
    Secondary safety output: minimum of the predicted 1..horizon trajectory.
    """

    horizon = int(horizon_hours)
    if not 1 <= horizon <= 24:
        raise ValueError("horizon_hours must be between 1 and 24")

    data = normalise_historical(frame)
    scored = compute_score_components(data, config=score_config)
    features = build_feature_frame(data)
    groups = data["hive_id"]

    targets = pd.DataFrame(index=data.index)
    for hour in range(1, horizon + 1):
        targets[f"score_t_plus_{hour}h"] = (
            scored["brood_health_score"].groupby(groups, sort=False).shift(-hour)
        )

    target_timestamp = data["timestamp"].groupby(groups, sort=False).shift(-horizon)
    current_score = scored["brood_health_score"]
    exact_future_score = targets[f"score_t_plus_{horizon}h"]
    minimum_future_score = targets.min(axis=1, skipna=False)

    metadata = data[["hive_id", "timestamp"]].copy()
    metadata["target_timestamp"] = target_timestamp
    metadata["current_score"] = current_score
    metadata["current_level_code"] = health_level_code(current_score)
    metadata["exact_future_score"] = exact_future_score
    metadata["exact_future_level_code"] = health_level_code(
        exact_future_score.fillna(current_score)
    )
    metadata["minimum_future_score"] = minimum_future_score
    metadata["minimum_future_level_code"] = health_level_code(
        minimum_future_score.fillna(current_score)
    )
    metadata["exact_score_drop"] = current_score - exact_future_score
    metadata["minimum_score_drop"] = current_score - minimum_future_score
    metadata["transition_window"] = (
        metadata["current_level_code"] != metadata["exact_future_level_code"]
    ) | (metadata["exact_score_drop"].abs() >= 10.0)
    metadata["deterioration_event"] = (
        metadata["exact_future_level_code"] < metadata["current_level_code"]
    ) | (metadata["exact_score_drop"] >= 10.0)
    metadata["history_hours"] = data.groupby("hive_id", sort=False).cumcount()

    if TARGET_COLUMN in data.columns:
        metadata["observed_health_at_horizon"] = (
            pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
            .groupby(groups, sort=False)
            .shift(-horizon)
        )
    else:
        metadata["observed_health_at_horizon"] = np.nan

    keep = (
        targets.notna().all(axis=1)
        & metadata["target_timestamp"].notna()
        & (metadata["history_hours"] >= MINIMUM_TRAINING_HISTORY_HOURS)
    )
    x = features.loc[keep].reset_index(drop=True)
    y = targets.loc[keep].astype(float).reset_index(drop=True)
    meta = metadata.loc[keep].reset_index(drop=True)

    feature_columns = list(x.columns)
    forbidden_fragments = (
        "brood_health",
        "healthy_1",
        "target",
        "future_score",
        "hive_id",
        "timestamp",
    )
    leaked = [
        column
        for column in feature_columns
        if any(fragment in column.lower() for fragment in forbidden_fragments)
    ]
    if leaked:
        raise RuntimeError(
            f"Forbidden columns entered the brood-health feature matrix: {sorted(leaked)}"
        )
    return x, y, meta, feature_columns


def build_latest_inference_rows(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = normalise_historical(frame)
    features = build_feature_frame(data)
    latest_positions = data.groupby("hive_id", sort=False)["timestamp"].idxmax()
    latest_meta = data.loc[latest_positions, ["hive_id", "timestamp", *SENSORS]].copy()
    latest_features = features.loc[latest_positions].reindex(columns=list(feature_columns))
    latest_features.index = latest_meta.index
    return latest_features.reset_index(drop=True), latest_meta.reset_index(drop=True)
