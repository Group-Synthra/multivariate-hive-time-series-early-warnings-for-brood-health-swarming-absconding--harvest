from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multivari.common.io import read_table, write_parquet
from multivari.common.schema import (
    HIVE_COLUMN,
    SENSOR_SANITY_BOUNDS,
    TIMESTAMP_COLUMN,
)
from multivari.common.splitting import assign_chronological_splits
from multivari.common.targets import make_future_event_target

from .config import AbscondingSettings
from .events import build_event_episodes

EXTERNAL_SENSOR_COLUMNS = (
    "external_temperature_c",
    "external_humidity_pct",
)
CORE_SENSOR_COLUMNS = (
    "temperature_c",
    "humidity_pct",
    "co2_ppm",
    "weight_kg",
)


@dataclass(frozen=True)
class AbscondingDataProfile:
    source_path: str
    rows_read: int
    rows_clean: int
    hives: int
    analysis_start: str
    analysis_end: str
    duplicate_hive_timestamps_removed: int
    invalid_timestamps_removed: int
    source_active_event_rows: int
    event_onset_markers: int
    merged_event_episodes: int
    derived_24h_positive_rows: int
    derived_24h_positive_rate: float
    source_72h_positive_rows: int | None
    derived_72h_positive_rows: int
    source_derived_72h_agreement: float | None
    median_sampling_minutes: float | None
    non_hourly_interval_fraction: float | None
    missing_values_after_cleaning: dict[str, int]
    sensor_ranges: dict[str, dict[str, float | None]]
    split_summary: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_absconding_data_pipeline(
    *,
    input_path: str | Path,
    backend_root: str | Path,
    config_path: str | Path,
) -> AbscondingDataProfile:
    """Prepare only the Absconding dataset without touching shared module data."""
    root = Path(backend_root).resolve()
    settings = AbscondingSettings.from_yaml(config_path)
    source = Path(input_path)
    if not source.is_absolute():
        source = root / source
    if not source.is_file():
        raise FileNotFoundError(f"Absconding source dataset was not found: {source}")

    raw = read_table(source)
    clean, diagnostics = clean_absconding_source(raw, settings)

    clean_path = _resolve(root, settings.data_clean_path)
    manifest_path = _resolve(root, settings.data_manifest_path)
    profile_path = _resolve(root, settings.data_profile_path)

    write_parquet(clean, clean_path)
    manifest = assign_chronological_splits(
        clean,
        train_fraction=settings.train_fraction,
        validation_fraction=settings.validation_fraction,
        boundary_gap_hours=max(
            settings.boundary_gap_hours,
            settings.prediction_horizon_hours,
        ),
    )
    write_parquet(manifest, manifest_path)

    profile = build_absconding_data_profile(
        clean,
        manifest,
        raw=raw,
        source_path=source,
        diagnostics=diagnostics,
        settings=settings,
    )
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return profile


def clean_absconding_source(
    raw: pd.DataFrame,
    settings: AbscondingSettings,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map the supplied historical file to the Absconding deployment contract.

    Only model-available sensor columns are retained. Labels belonging to other modules and
    the supplied future target are deliberately excluded from the model table to prevent
    accidental target leakage.
    """
    mapping = {
        settings.source_timestamp_column: TIMESTAMP_COLUMN,
        settings.source_hive_column: HIVE_COLUMN,
        settings.source_temperature_column: "temperature_c",
        settings.source_humidity_column: "humidity_pct",
        settings.source_co2_column: "co2_ppm",
        settings.source_weight_column: "weight_kg",
        settings.source_external_temperature_column: "external_temperature_c",
        settings.source_external_humidity_column: "external_humidity_pct",
        settings.source_event_column: settings.active_event_column,
    }
    required_source = {
        settings.source_timestamp_column,
        settings.source_hive_column,
        settings.source_temperature_column,
        settings.source_humidity_column,
        settings.source_co2_column,
        settings.source_weight_column,
        settings.source_event_column,
    }
    missing = sorted(required_source - set(raw.columns))
    if missing:
        raise ValueError(
            "The separate Absconding dataset is missing required columns: "
            f"{missing}. Update config/absconding.yaml if the names differ."
        )

    available_mapping = {source: target for source, target in mapping.items() if source in raw}
    frame = raw[list(available_mapping)].rename(columns=available_mapping).copy()
    rows_read = len(frame)

    timestamps = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
    invalid_timestamp_count = int(timestamps.isna().sum())
    frame[TIMESTAMP_COLUMN] = timestamps
    frame[HIVE_COLUMN] = frame[HIVE_COLUMN].astype("string").str.strip()
    frame = frame.dropna(subset=[TIMESTAMP_COLUMN, HIVE_COLUMN])
    frame = frame.loc[frame[HIVE_COLUMN].ne("")].copy()

    numeric_columns = [
        *CORE_SENSOR_COLUMNS,
        *[column for column in EXTERNAL_SENSOR_COLUMNS if column in frame],
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame[settings.active_event_column] = (
        pd.to_numeric(frame[settings.active_event_column], errors="coerce")
        .fillna(0)
        .gt(0)
        .astype("int8")
    )

    frame = frame.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN])
    duplicate_count = int(frame.duplicated([HIVE_COLUMN, TIMESTAMP_COLUMN]).sum())
    frame = frame.drop_duplicates([HIVE_COLUMN, TIMESTAMP_COLUMN], keep="last").copy()

    sanity_bounds = {
        **SENSOR_SANITY_BOUNDS,
        "external_temperature_c": (-50.0, 60.0),
        "external_humidity_pct": (0.0, 100.0),
    }
    for column in numeric_columns:
        lower, upper = sanity_bounds[column]
        invalid = frame[column].lt(lower) | frame[column].gt(upper)
        frame.loc[invalid, column] = np.nan

    # Past-only filling prevents future observations from leaking into earlier feature rows.
    for column in numeric_columns:
        frame[column] = frame.groupby(HIVE_COLUMN, sort=False)[column].ffill(
            limit=settings.interpolation_limit_hours
        )

    previous_active = frame.groupby(HIVE_COLUMN, sort=False)[settings.active_event_column].shift(
        fill_value=0
    )
    frame[settings.event_column] = (
        frame[settings.active_event_column].eq(1) & previous_active.eq(0)
    ).astype("int8")
    frame["dataset_source"] = "separate_absconding_dataset"

    source_target = None
    if settings.source_precomputed_target_column in raw:
        target_lookup = raw[
            [
                settings.source_timestamp_column,
                settings.source_hive_column,
                settings.source_precomputed_target_column,
            ]
        ].copy()
        target_lookup[TIMESTAMP_COLUMN] = pd.to_datetime(
            target_lookup[settings.source_timestamp_column], errors="coerce"
        )
        target_lookup[HIVE_COLUMN] = (
            target_lookup[settings.source_hive_column].astype("string").str.strip()
        )
        target_lookup["source_72h_target"] = (
            pd.to_numeric(target_lookup[settings.source_precomputed_target_column], errors="coerce")
            .fillna(0)
            .gt(0)
            .astype("int8")
        )
        source_target = target_lookup[
            [HIVE_COLUMN, TIMESTAMP_COLUMN, "source_72h_target"]
        ].drop_duplicates([HIVE_COLUMN, TIMESTAMP_COLUMN], keep="last")

    output_columns = [
        TIMESTAMP_COLUMN,
        HIVE_COLUMN,
        *CORE_SENSOR_COLUMNS,
        *[column for column in EXTERNAL_SENSOR_COLUMNS if column in frame],
        settings.active_event_column,
        settings.event_column,
        "dataset_source",
    ]
    clean = frame[output_columns].reset_index(drop=True)
    diagnostics = {
        "rows_read": rows_read,
        "invalid_timestamps_removed": invalid_timestamp_count,
        "duplicate_hive_timestamps_removed": duplicate_count,
        "source_target": source_target,
    }
    return clean, diagnostics


def build_absconding_data_profile(
    clean: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    raw: pd.DataFrame,
    source_path: Path,
    diagnostics: dict[str, Any],
    settings: AbscondingSettings,
) -> AbscondingDataProfile:
    target_24 = make_future_event_target(
        clean,
        event_column=settings.event_column,
        horizon_hours=settings.prediction_horizon_hours,
        output_column=settings.target_column,
    )
    target_72 = make_future_event_target(
        clean,
        event_column=settings.event_column,
        horizon_hours=72,
        output_column="derived_absconding_within_72h",
    )
    merged = build_event_episodes(
        clean,
        event_column=settings.event_column,
        merge_gap_hours=settings.event_merge_gap_hours,
    )

    source_target = diagnostics.get("source_target")
    source_72_rows: int | None = None
    agreement: float | None = None
    if isinstance(source_target, pd.DataFrame) and not source_target.empty:
        compare = target_72[[HIVE_COLUMN, TIMESTAMP_COLUMN, "derived_absconding_within_72h"]].merge(
            source_target,
            on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
            how="inner",
            validate="one_to_one",
        )
        compare = compare.dropna(subset=["derived_absconding_within_72h"])
        source_72_rows = int(
            pd.to_numeric(raw[settings.source_precomputed_target_column], errors="coerce")
            .fillna(0)
            .gt(0)
            .sum()
        )
        if not compare.empty:
            agreement = round(
                float(
                    compare["derived_absconding_within_72h"]
                    .astype("int8")
                    .eq(compare["source_72h_target"])
                    .mean()
                ),
                8,
            )

    intervals = (
        clean.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN])
        .groupby(HIVE_COLUMN, sort=False)[TIMESTAMP_COLUMN]
        .diff()
        .dt.total_seconds()
        .div(60)
        .dropna()
    )
    median_minutes = round(float(intervals.median()), 4) if not intervals.empty else None
    non_hourly_fraction = (
        round(float(1.0 - intervals.between(59.5, 60.5).mean()), 8) if not intervals.empty else None
    )

    split_with_events = manifest.merge(
        clean[[HIVE_COLUMN, TIMESTAMP_COLUMN, settings.event_column]],
        on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        how="left",
        validate="one_to_one",
    )
    split_summary = []
    for split in ("train", "validation", "test"):
        group = split_with_events.loc[split_with_events["split"].eq(split)]
        split_summary.append(
            {
                "split": split,
                "rows": len(group),
                "boundary_gap_rows": int(group["is_boundary_gap"].sum()),
                "event_onset_markers": int(group[settings.event_column].sum()),
            }
        )

    sensor_columns = [
        *CORE_SENSOR_COLUMNS,
        *[column for column in EXTERNAL_SENSOR_COLUMNS if column in clean],
    ]
    ranges: dict[str, dict[str, float | None]] = {}
    for column in sensor_columns:
        series = clean[column]
        ranges[column] = {
            "min": _finite_or_none(series.min()),
            "max": _finite_or_none(series.max()),
            "mean": _finite_or_none(series.mean()),
        }

    positive_24 = int(target_24[settings.target_column].fillna(0).sum())
    valid_24 = target_24[settings.target_column].notna()
    return AbscondingDataProfile(
        source_path=str(source_path),
        rows_read=int(diagnostics["rows_read"]),
        rows_clean=len(clean),
        hives=int(clean[HIVE_COLUMN].nunique()),
        analysis_start=clean[TIMESTAMP_COLUMN].min().isoformat(),
        analysis_end=clean[TIMESTAMP_COLUMN].max().isoformat(),
        duplicate_hive_timestamps_removed=int(diagnostics["duplicate_hive_timestamps_removed"]),
        invalid_timestamps_removed=int(diagnostics["invalid_timestamps_removed"]),
        source_active_event_rows=int(clean[settings.active_event_column].sum()),
        event_onset_markers=int(clean[settings.event_column].sum()),
        merged_event_episodes=len(merged),
        derived_24h_positive_rows=positive_24,
        derived_24h_positive_rate=round(
            float(target_24.loc[valid_24, settings.target_column].mean()), 8
        ),
        source_72h_positive_rows=source_72_rows,
        derived_72h_positive_rows=int(target_72["derived_absconding_within_72h"].fillna(0).sum()),
        source_derived_72h_agreement=agreement,
        median_sampling_minutes=median_minutes,
        non_hourly_interval_fraction=non_hourly_fraction,
        missing_values_after_cleaning={
            column: int(clean[column].isna().sum()) for column in sensor_columns
        },
        sensor_ranges=ranges,
        split_summary=split_summary,
    )


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6) if np.isfinite(number) else None
