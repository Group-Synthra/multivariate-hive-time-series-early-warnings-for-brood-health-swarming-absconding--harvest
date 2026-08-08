from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

HIVE_COLUMN = "hive_id"
TIMESTAMP_COLUMN = "timestamp"
SPLIT_COLUMN = "split"

SENSOR_COLUMNS = {
    "weight": "weight_kg",
    "temperature": "temperature_c",
    "humidity": "humidity_pct",
    "co2": "co2_ppm",
}

BANNED_MODEL_COLUMNS = {
    "brood_health_healthy_1",
    "swarming_happened_1",
    "absconding_happened_1",
    "honey_harvested_1",
    "harvest_event_start_1",
    "harvest_reviewed_event_start_1",
    "harvest_event_id",
    "is_boundary_gap",
    "is_post_event_recovery_gap",
}


@dataclass(frozen=True)
class FeatureDefinition:
    feature_name: str
    category: str
    sensor: str
    lookback_hours: int
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "category": self.category,
            "sensor": self.sensor,
            "lookback_hours": self.lookback_hours,
            "live_available": True,
            "notes": self.notes,
        }


def _resolve_path(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    frame_name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _validate_unique_keys(frame: pd.DataFrame) -> None:
    duplicated = frame.duplicated(
        subset=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        keep=False,
    )
    if duplicated.any():
        examples = (
            frame.loc[
                duplicated,
                [HIVE_COLUMN, TIMESTAMP_COLUMN],
            ]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"Feature source contains duplicate hive/timestamp keys. Examples: {examples}"
        )


def _add_contiguous_segment_id(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    hourly_difference = (
        result.groupby(HIVE_COLUMN, sort=False)[TIMESTAMP_COLUMN]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )
    starts_new_segment = hourly_difference.isna() | hourly_difference.ne(1.0)
    result["_segment_id"] = (
        starts_new_segment.groupby(
            result[HIVE_COLUMN],
            sort=False,
        )
        .cumsum()
        .astype("int32")
    )
    result["_hours_since_previous"] = hourly_difference
    return result


def _rolling_stat(
    frame: pd.DataFrame,
    *,
    column: str,
    window: int,
    statistic: str,
) -> pd.Series:
    grouped = frame.groupby(
        [HIVE_COLUMN, "_segment_id"],
        sort=False,
    )[column]
    rolling = grouped.rolling(
        window=window,
        min_periods=window,
    )

    if statistic == "mean":
        values = rolling.mean()
    elif statistic == "std":
        values = rolling.std(ddof=1)
    elif statistic == "min":
        values = rolling.min()
    elif statistic == "max":
        values = rolling.max()
    else:
        raise ValueError(f"Unsupported rolling statistic: {statistic}")

    return values.reset_index(level=[0, 1], drop=True).reindex(frame.index)


def _group_shift(
    frame: pd.DataFrame,
    *,
    column: str,
    periods: int,
) -> pd.Series:
    return frame.groupby(
        [HIVE_COLUMN, "_segment_id"],
        sort=False,
    )[column].shift(periods)


def _append_feature(
    feature_frame: pd.DataFrame,
    manifest: list[FeatureDefinition],
    *,
    name: str,
    values: pd.Series | np.ndarray,
    category: str,
    sensor: str,
    lookback_hours: int,
    notes: str,
) -> None:
    feature_frame[name] = values
    manifest.append(
        FeatureDefinition(
            feature_name=name,
            category=category,
            sensor=sensor,
            lookback_hours=lookback_hours,
            notes=notes,
        )
    )


def _build_weight_features(
    frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    manifest: list[FeatureDefinition],
    *,
    windows: list[int],
    delta_hours: list[int],
    trend_hours: list[int],
) -> None:
    column = SENSOR_COLUMNS["weight"]

    _append_feature(
        feature_frame,
        manifest,
        name="weight_kg_current",
        values=frame[column],
        category="current",
        sensor="weight",
        lookback_hours=0,
        notes="Current hive weight.",
    )

    for hours in delta_hours:
        lagged = _group_shift(
            frame,
            column=column,
            periods=hours,
        )
        _append_feature(
            feature_frame,
            manifest,
            name=f"weight_delta_{hours}h_kg",
            values=frame[column] - lagged,
            category="change",
            sensor="weight",
            lookback_hours=hours,
            notes=(f"Current weight minus weight {hours} hours earlier."),
        )

    rolling_cache: dict[tuple[int, str], pd.Series] = {}
    for window in windows:
        for statistic in ("mean", "std", "min", "max"):
            rolling_cache[(window, statistic)] = _rolling_stat(
                frame,
                column=column,
                window=window,
                statistic=statistic,
            )

        _append_feature(
            feature_frame,
            manifest,
            name=f"weight_mean_{window}h_kg",
            values=rolling_cache[(window, "mean")],
            category="rolling",
            sensor="weight",
            lookback_hours=window,
            notes=f"Rolling mean over {window} hourly observations.",
        )
        _append_feature(
            feature_frame,
            manifest,
            name=f"weight_std_{window}h_kg",
            values=rolling_cache[(window, "std")],
            category="rolling",
            sensor="weight",
            lookback_hours=window,
            notes=f"Rolling standard deviation over {window} hours.",
        )
        _append_feature(
            feature_frame,
            manifest,
            name=f"weight_range_{window}h_kg",
            values=(rolling_cache[(window, "max")] - rolling_cache[(window, "min")]),
            category="rolling",
            sensor="weight",
            lookback_hours=window,
            notes=f"Rolling weight range over {window} hours.",
        )

    for hours in trend_hours:
        lagged = _group_shift(
            frame,
            column=column,
            periods=hours,
        )
        _append_feature(
            feature_frame,
            manifest,
            name=f"weight_trend_{hours}h_kg_per_hour",
            values=(frame[column] - lagged) / hours,
            category="trend",
            sensor="weight",
            lookback_hours=hours,
            notes=(
                f"Endpoint rate of change using only the current and {hours}-hour lagged weight."
            ),
        )

    for window in (24, 72, 168):
        if (window, "max") not in rolling_cache:
            continue
        rolling_max = rolling_cache[(window, "max")]
        _append_feature(
            feature_frame,
            manifest,
            name=f"weight_distance_from_max_{window}h_kg",
            values=rolling_max - frame[column],
            category="domain",
            sensor="weight",
            lookback_hours=window,
            notes=(f"Distance below the rolling {window}-hour maximum."),
        )

    for window in (72, 168):
        if (window, "max") not in rolling_cache:
            continue
        rolling_max = rolling_cache[(window, "max")]
        safe_max = rolling_max.where(rolling_max.abs().gt(1e-9))
        _append_feature(
            feature_frame,
            manifest,
            name=f"weight_relative_to_max_{window}h",
            values=frame[column] / safe_max,
            category="domain",
            sensor="weight",
            lookback_hours=window,
            notes=(f"Current weight divided by the rolling {window}-hour maximum."),
        )


def _build_environmental_features(
    frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    manifest: list[FeatureDefinition],
    *,
    sensor_name: str,
    windows: list[int],
    delta_hours: list[int],
    trend_hours: list[int],
) -> None:
    column = SENSOR_COLUMNS[sensor_name]

    _append_feature(
        feature_frame,
        manifest,
        name=f"{column}_current",
        values=frame[column],
        category="current",
        sensor=sensor_name,
        lookback_hours=0,
        notes=f"Current {sensor_name} reading.",
    )

    for hours in delta_hours:
        lagged = _group_shift(
            frame,
            column=column,
            periods=hours,
        )
        _append_feature(
            feature_frame,
            manifest,
            name=f"{column}_delta_{hours}h",
            values=frame[column] - lagged,
            category="change",
            sensor=sensor_name,
            lookback_hours=hours,
            notes=(f"Current reading minus the reading {hours} hours earlier."),
        )

    for window in windows:
        rolling_mean = _rolling_stat(
            frame,
            column=column,
            window=window,
            statistic="mean",
        )
        rolling_std = _rolling_stat(
            frame,
            column=column,
            window=window,
            statistic="std",
        )
        rolling_min = _rolling_stat(
            frame,
            column=column,
            window=window,
            statistic="min",
        )
        rolling_max = _rolling_stat(
            frame,
            column=column,
            window=window,
            statistic="max",
        )

        _append_feature(
            feature_frame,
            manifest,
            name=f"{column}_mean_{window}h",
            values=rolling_mean,
            category="rolling",
            sensor=sensor_name,
            lookback_hours=window,
            notes=f"Rolling mean over {window} hourly observations.",
        )
        _append_feature(
            feature_frame,
            manifest,
            name=f"{column}_std_{window}h",
            values=rolling_std,
            category="rolling",
            sensor=sensor_name,
            lookback_hours=window,
            notes=f"Rolling standard deviation over {window} hours.",
        )

        if window == 24:
            _append_feature(
                feature_frame,
                manifest,
                name=f"{column}_range_{window}h",
                values=rolling_max - rolling_min,
                category="rolling",
                sensor=sensor_name,
                lookback_hours=window,
                notes=f"Rolling range over {window} hours.",
            )

    for hours in trend_hours:
        lagged = _group_shift(
            frame,
            column=column,
            periods=hours,
        )
        _append_feature(
            feature_frame,
            manifest,
            name=f"{column}_trend_{hours}h_per_hour",
            values=(frame[column] - lagged) / hours,
            category="trend",
            sensor=sensor_name,
            lookback_hours=hours,
            notes=(f"Endpoint rate of change using the current and {hours}-hour lagged reading."),
        )


def build_reviewed_feature_dataset(
    history: pd.DataFrame,
    modelling: pd.DataFrame | None = None,
    *,
    target_column: str,
    minimum_history_hours: int,
    weight_windows_hours: list[int],
    environmental_windows_hours: list[int],
    weight_delta_hours: list[int],
    environmental_delta_hours: list[int],
    weight_trend_hours: list[int],
    environmental_trend_hours: list[int],
    co2_flatline_std_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Build live-compatible features from continuous sensor history.

    Feature calculations use the complete cleaned sensor timeline. The
    resulting feature rows are then joined to the already approved modelling
    rows. This keeps split-boundary and post-event purge rows out of model
    fitting without incorrectly treating those removed rows as missing sensor
    history.
    """
    history_required = {
        HIVE_COLUMN,
        TIMESTAMP_COLUMN,
        *SENSOR_COLUMNS.values(),
    }
    _require_columns(
        history,
        history_required,
        frame_name="Continuous sensor history",
    )

    modelling_frame = history.copy() if modelling is None else modelling.copy()
    modelling_required = {
        HIVE_COLUMN,
        TIMESTAMP_COLUMN,
        SPLIT_COLUMN,
        target_column,
    }
    _require_columns(
        modelling_frame,
        modelling_required,
        frame_name="Reviewed modelling dataset",
    )

    if minimum_history_hours <= 0:
        raise ValueError("minimum_history_hours must be greater than zero")

    frame = history.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        frame[TIMESTAMP_COLUMN],
        errors="raise",
    )
    frame = frame.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)
    _validate_unique_keys(frame)

    modelling_frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        modelling_frame[TIMESTAMP_COLUMN],
        errors="raise",
    )
    modelling_frame = modelling_frame.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN]).reset_index(
        drop=True
    )
    _validate_unique_keys(modelling_frame)

    key_check = modelling_frame[[HIVE_COLUMN, TIMESTAMP_COLUMN]].merge(
        frame[[HIVE_COLUMN, TIMESTAMP_COLUMN]],
        on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    unmatched_key_count = int(key_check["_merge"].ne("both").sum())
    if unmatched_key_count:
        raise ValueError(
            "Some modelling rows do not exist in the continuous "
            f"sensor history: {unmatched_key_count}"
        )

    frame = _add_contiguous_segment_id(frame)

    gap_rows = int(
        frame["_hours_since_previous"].notna().mul(frame["_hours_since_previous"].ne(1.0)).sum()
    )
    segment_count = int(frame[[HIVE_COLUMN, "_segment_id"]].drop_duplicates().shape[0])

    feature_frame = pd.DataFrame(index=frame.index)
    manifest: list[FeatureDefinition] = []

    _build_weight_features(
        frame,
        feature_frame,
        manifest,
        windows=weight_windows_hours,
        delta_hours=weight_delta_hours,
        trend_hours=weight_trend_hours,
    )

    for sensor_name in ("temperature", "humidity", "co2"):
        _build_environmental_features(
            frame,
            feature_frame,
            manifest,
            sensor_name=sensor_name,
            windows=environmental_windows_hours,
            delta_hours=environmental_delta_hours,
            trend_hours=environmental_trend_hours,
        )

    hour = frame[TIMESTAMP_COLUMN].dt.hour.to_numpy()
    day_of_week = frame[TIMESTAMP_COLUMN].dt.dayofweek.to_numpy()

    _append_feature(
        feature_frame,
        manifest,
        name="hour_sin",
        values=np.sin(2 * np.pi * hour / 24),
        category="time",
        sensor="calendar",
        lookback_hours=0,
        notes="Cyclical hour-of-day sine component.",
    )
    _append_feature(
        feature_frame,
        manifest,
        name="hour_cos",
        values=np.cos(2 * np.pi * hour / 24),
        category="time",
        sensor="calendar",
        lookback_hours=0,
        notes="Cyclical hour-of-day cosine component.",
    )
    _append_feature(
        feature_frame,
        manifest,
        name="day_of_week_sin",
        values=np.sin(2 * np.pi * day_of_week / 7),
        category="time",
        sensor="calendar",
        lookback_hours=0,
        notes="Cyclical day-of-week sine component.",
    )
    _append_feature(
        feature_frame,
        manifest,
        name="day_of_week_cos",
        values=np.cos(2 * np.pi * day_of_week / 7),
        category="time",
        sensor="calendar",
        lookback_hours=0,
        notes="Cyclical day-of-week cosine component.",
    )

    co2_std_24 = feature_frame["co2_ppm_std_24h"]
    co2_std_72 = feature_frame["co2_ppm_std_72h"]
    _append_feature(
        feature_frame,
        manifest,
        name="co2_flatline_24h_1",
        values=(co2_std_24.le(co2_flatline_std_threshold).astype("float64")),
        category="quality",
        sensor="co2",
        lookback_hours=24,
        notes=(
            "One when the rolling 24-hour CO2 standard deviation "
            "is at or below the configured flatline threshold."
        ),
    )
    _append_feature(
        feature_frame,
        manifest,
        name="co2_flatline_72h_1",
        values=(co2_std_72.le(co2_flatline_std_threshold).astype("float64")),
        category="quality",
        sensor="co2",
        lookback_hours=72,
        notes=(
            "One when the rolling 72-hour CO2 standard deviation "
            "is at or below the configured flatline threshold."
        ),
    )

    _append_feature(
        feature_frame,
        manifest,
        name="environmental_variability_24h",
        values=(
            feature_frame["temperature_c_std_24h"]
            + feature_frame["humidity_pct_std_24h"]
            + feature_frame["co2_ppm_std_24h"] / 100.0
        ),
        category="domain",
        sensor="environment",
        lookback_hours=24,
        notes=(
            "Unscaled environmental variability proxy using "
            "temperature, humidity and CO2 rolling standard deviations."
        ),
    )
    _append_feature(
        feature_frame,
        manifest,
        name="environmental_variability_72h",
        values=(
            feature_frame["temperature_c_std_72h"]
            + feature_frame["humidity_pct_std_72h"]
            + feature_frame["co2_ppm_std_72h"] / 100.0
        ),
        category="domain",
        sensor="environment",
        lookback_hours=72,
        notes=("Unscaled environmental variability proxy over 72 hours."),
    )

    feature_frame = feature_frame.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    feature_columns = list(feature_frame.columns)
    prohibited = sorted(set(feature_columns).intersection(BANNED_MODEL_COLUMNS))
    if prohibited:
        raise ValueError(f"Leakage columns were included in the feature frame: {prohibited}")

    complete_history_mask = (
        frame.groupby(
            [HIVE_COLUMN, "_segment_id"],
            sort=False,
        )
        .cumcount()
        .ge(minimum_history_hours - 1)
    )
    finite_mask = feature_frame.notna().all(axis=1)
    keep_history_mask = complete_history_mask & finite_mask

    feature_lookup = pd.concat(
        [
            frame.loc[
                keep_history_mask,
                [HIVE_COLUMN, TIMESTAMP_COLUMN],
            ].reset_index(drop=True),
            feature_frame.loc[
                keep_history_mask,
                feature_columns,
            ]
            .astype("float32")
            .reset_index(drop=True),
        ],
        axis=1,
    )

    metadata_columns = [
        TIMESTAMP_COLUMN,
        HIVE_COLUMN,
        SPLIT_COLUMN,
        target_column,
    ]
    output = modelling_frame[metadata_columns].merge(
        feature_lookup,
        on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        how="left",
        validate="one_to_one",
    )

    missing_feature_mask = output[feature_columns].isna().any(axis=1)
    output = output.loc[~missing_feature_mask].copy()

    output = output.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)

    if output.empty:
        raise ValueError("No rows remained after feature-history filtering.")

    manifest_frame = pd.DataFrame([definition.as_dict() for definition in manifest])

    source_positive_rows = int(modelling_frame[target_column].sum())
    output_positive_rows = int(output[target_column].sum())

    split_balance = (
        output.groupby(
            [SPLIT_COLUMN, target_column],
            observed=True,
        )
        .size()
        .rename("rows")
        .reset_index()
        .to_dict(orient="records")
    )

    audit: dict[str, Any] = {
        "history_rows": len(frame),
        "source_rows": len(modelling_frame),
        "output_rows": len(output),
        "rows_removed_for_history_or_missing_features": int(len(modelling_frame) - len(output)),
        "feature_count": len(feature_columns),
        "minimum_history_hours": minimum_history_hours,
        "contiguous_segment_count": segment_count,
        "detected_non_hourly_gaps": gap_rows,
        "modelling_rows_without_history_key": unmatched_key_count,
        "source_positive_rows": source_positive_rows,
        "output_positive_rows": output_positive_rows,
        "positive_rows_removed": (source_positive_rows - output_positive_rows),
        "output_positive_rate": float(output[target_column].mean()),
        "feature_columns": feature_columns,
        "split_balance": split_balance,
        "leakage_columns_present": prohibited,
        "history_policy": (
            "Features are calculated from the complete cleaned sensor "
            "timeline and then joined to approved modelling rows. Purged "
            "rows are excluded as samples but remain available as past "
            "sensor history, matching live inference."
        ),
        "warning": (
            "Features use only current and past observations. "
            "The dataset still contains only 12 probable pseudo-events; "
            "feature selection and model evaluation must be conservative."
        ),
    }

    return output, manifest_frame, audit


def run_reviewed_features_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    settings = config["reviewed_features"]
    target_column = config["reviewed_target"]["output_column"]

    history_path = _resolve_path(
        root,
        config["dataset"]["clean_data_path"],
    )
    source_path = _resolve_path(
        root,
        settings["source_dataset_path"],
    )
    output_path = _resolve_path(
        root,
        settings["output_dataset_path"],
    )
    manifest_path = _resolve_path(
        root,
        settings["feature_manifest_path"],
    )
    audit_path = _resolve_path(
        root,
        settings["audit_path"],
    )

    history = pd.read_parquet(history_path)
    source = pd.read_parquet(source_path)

    output, manifest, audit = build_reviewed_feature_dataset(
        history,
        source,
        target_column=target_column,
        minimum_history_hours=int(settings["minimum_history_hours"]),
        weight_windows_hours=[int(value) for value in settings["weight_windows_hours"]],
        environmental_windows_hours=[
            int(value) for value in settings["environmental_windows_hours"]
        ],
        weight_delta_hours=[int(value) for value in settings["weight_delta_hours"]],
        environmental_delta_hours=[int(value) for value in settings["environmental_delta_hours"]],
        weight_trend_hours=[int(value) for value in settings["weight_trend_hours"]],
        environmental_trend_hours=[int(value) for value in settings["environmental_trend_hours"]],
        co2_flatline_std_threshold=float(settings["co2_flatline_std_threshold"]),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_parquet(output_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    audit_path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    return {
        **audit,
        "history_dataset_path": str(history_path),
        "output_dataset_path": str(output_path),
        "feature_manifest_path": str(manifest_path),
        "audit_path": str(audit_path),
    }
