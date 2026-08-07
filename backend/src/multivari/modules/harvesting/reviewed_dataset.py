from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

HIVE_COLUMN = "hive_id"
TIMESTAMP_COLUMN = "timestamp"


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
        raise ValueError(
            f"{frame_name} is missing required columns: {missing}"
        )


def _validate_unique_keys(
    frame: pd.DataFrame,
    *,
    frame_name: str,
) -> None:
    duplicates = frame.duplicated(
        subset=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        keep=False,
    )
    if duplicates.any():
        examples = (
            frame.loc[
                duplicates,
                [HIVE_COLUMN, TIMESTAMP_COLUMN],
            ]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{frame_name} contains duplicate hive/timestamp keys. "
            f"Examples: {examples}"
        )


def make_future_reviewed_event_target(
    rows: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon_hours: int,
    output_column: str,
) -> pd.DataFrame:
    """
    Label each row when a reviewed event begins in (t, t + horizon].

    The current timestamp is excluded. Rows whose complete future horizon
    is not observable are assigned a missing target.

    Datetimes are explicitly converted to nanosecond NumPy arrays so the
    calculation remains correct across pandas versions that internally use
    different datetime resolutions.
    """
    if horizon_hours <= 0:
        raise ValueError(
            "horizon_hours must be greater than zero"
        )

    _require_columns(
        rows,
        {HIVE_COLUMN, TIMESTAMP_COLUMN},
        frame_name="Sensor rows",
    )
    _require_columns(
        events,
        {HIVE_COLUMN, "event_start"},
        frame_name="Reviewed event table",
    )

    result = rows.copy()
    result[TIMESTAMP_COLUMN] = pd.to_datetime(
        result[TIMESTAMP_COLUMN],
        errors="raise",
    )
    result = result.sort_values(
        [HIVE_COLUMN, TIMESTAMP_COLUMN]
    ).reset_index(drop=True)

    event_frame = events[
        [HIVE_COLUMN, "event_start"]
    ].copy()
    event_frame["event_start"] = pd.to_datetime(
        event_frame["event_start"],
        errors="raise",
    )
    event_frame = event_frame.drop_duplicates().sort_values(
        [HIVE_COLUMN, "event_start"]
    )

    event_times_by_hive = {
        hive_id: group["event_start"].to_numpy(
            dtype="datetime64[ns]"
        )
        for hive_id, group in event_frame.groupby(
            HIVE_COLUMN,
            sort=False,
        )
    }

    horizon = np.timedelta64(
        horizon_hours,
        "h",
    )
    target = np.full(
        len(result),
        np.nan,
        dtype=float,
    )

    for hive_id, group in result.groupby(
        HIVE_COLUMN,
        sort=False,
    ):
        indices = group.index.to_numpy()
        times = group[TIMESTAMP_COLUMN].to_numpy(
            dtype="datetime64[ns]"
        )

        if len(times) == 0:
            continue

        complete_future_available = (
            times + horizon <= times[-1]
        )
        hive_event_times = event_times_by_hive.get(
            hive_id,
            np.array([], dtype="datetime64[ns]"),
        )

        if len(hive_event_times) == 0:
            target[
                indices[complete_future_available]
            ] = 0
            continue

        next_event_positions = np.searchsorted(
            hive_event_times,
            times,
            side="right",
        )
        has_next_event = (
            next_event_positions
            < len(hive_event_times)
        )

        next_event_times = np.full(
            len(times),
            np.datetime64("NaT", "ns"),
            dtype="datetime64[ns]",
        )
        next_event_times[has_next_event] = (
            hive_event_times[
                next_event_positions[has_next_event]
            ]
        )

        within_horizon = (
            has_next_event
            & (
                next_event_times
                <= times + horizon
            )
        )

        valid_indices = indices[
            complete_future_available
        ]
        target[valid_indices] = within_horizon[
            complete_future_available
        ].astype(np.int8)

    result[output_column] = target
    return result


def add_reviewed_event_columns(
    rows: pd.DataFrame,
    events: pd.DataFrame,
    *,
    indicator_column: str,
) -> pd.DataFrame:
    """Attach reviewed event-start indicators and event IDs."""
    event_keys = events[
        [
            HIVE_COLUMN,
            "event_start",
            "harvest_event_id",
        ]
    ].copy()
    event_keys = event_keys.rename(
        columns={"event_start": TIMESTAMP_COLUMN}
    )
    event_keys[TIMESTAMP_COLUMN] = pd.to_datetime(
        event_keys[TIMESTAMP_COLUMN],
        errors="raise",
    )

    duplicates = event_keys.duplicated(
        subset=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        keep=False,
    )
    if duplicates.any():
        raise ValueError(
            "Reviewed event table contains duplicate event timestamps."
        )

    result = rows.merge(
        event_keys,
        on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        how="left",
        validate="one_to_one",
    )
    result[indicator_column] = (
        result["harvest_event_id"]
        .notna()
        .astype("int8")
    )
    return result


def add_post_event_recovery_gap(
    rows: pd.DataFrame,
    events: pd.DataFrame,
    *,
    recovery_hours: int,
) -> pd.DataFrame:
    """
    Mark the event timestamp and the immediate post-event period.

    The interval is inclusive: an event at 10:00 with recovery_hours=24
    marks 10:00 through the following day's 10:00. The next hour is not
    marked.

    Datetimes are explicitly converted to nanosecond NumPy arrays so the
    calculation remains correct across pandas versions.
    """
    if recovery_hours < 0:
        raise ValueError(
            "recovery_hours cannot be negative"
        )

    result = rows.copy()
    result[TIMESTAMP_COLUMN] = pd.to_datetime(
        result[TIMESTAMP_COLUMN],
        errors="raise",
    )
    result["is_post_event_recovery_gap"] = False

    event_frame = events.copy()
    event_frame["event_start"] = pd.to_datetime(
        event_frame["event_start"],
        errors="raise",
    )

    event_times_by_hive = {
        hive_id: group["event_start"]
        .sort_values()
        .to_numpy(dtype="datetime64[ns]")
        for hive_id, group in event_frame.groupby(
            HIVE_COLUMN,
            sort=False,
        )
    }

    recovery_interval = np.timedelta64(
        recovery_hours,
        "h",
    )

    for hive_id, group in result.groupby(
        HIVE_COLUMN,
        sort=False,
    ):
        event_times = event_times_by_hive.get(
            hive_id,
            np.array([], dtype="datetime64[ns]"),
        )
        if len(event_times) == 0:
            continue

        indices = group.index.to_numpy()
        times = group[TIMESTAMP_COLUMN].to_numpy(
            dtype="datetime64[ns]"
        )

        previous_positions = (
            np.searchsorted(
                event_times,
                times,
                side="right",
            )
            - 1
        )
        has_previous = previous_positions >= 0

        in_recovery = np.zeros(
            len(times),
            dtype=bool,
        )

        if has_previous.any():
            elapsed = (
                times[has_previous]
                - event_times[
                    previous_positions[has_previous]
                ]
            )
            in_recovery[has_previous] = (
                (elapsed >= np.timedelta64(0, "h"))
                & (elapsed <= recovery_interval)
            )

        result.loc[
            indices[in_recovery],
            "is_post_event_recovery_gap",
        ] = True

    result["is_post_event_recovery_gap"] = (
        result["is_post_event_recovery_gap"]
        .astype(bool)
    )
    return result


def build_reviewed_harvest_dataset(
    common: pd.DataFrame,
    split_manifest: pd.DataFrame,
    reviewed_events: pd.DataFrame,
    *,
    horizon_hours: int,
    target_column: str,
    event_indicator_column: str,
    post_event_recovery_hours: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Build the leakage-aware dataset from manually reviewed events."""
    _require_columns(
        common,
        {HIVE_COLUMN, TIMESTAMP_COLUMN},
        frame_name="Common cleaned dataset",
    )
    _require_columns(
        split_manifest,
        {
            HIVE_COLUMN,
            TIMESTAMP_COLUMN,
            "split",
        },
        frame_name="Split manifest",
    )
    _require_columns(
        reviewed_events,
        {
            HIVE_COLUMN,
            "event_start",
            "harvest_event_id",
        },
        frame_name="Reviewed event table",
    )

    common_frame = common.copy()
    common_frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        common_frame[TIMESTAMP_COLUMN],
        errors="raise",
    )
    _validate_unique_keys(
        common_frame,
        frame_name="Common cleaned dataset",
    )

    manifest = split_manifest.copy()
    manifest[TIMESTAMP_COLUMN] = pd.to_datetime(
        manifest[TIMESTAMP_COLUMN],
        errors="raise",
    )
    _validate_unique_keys(
        manifest,
        frame_name="Split manifest",
    )

    events = reviewed_events.copy()
    events["event_start"] = pd.to_datetime(
        events["event_start"],
        errors="raise",
    )
    if events.empty:
        raise ValueError(
            "Reviewed event table is empty. "
            "At least one probable harvest event is required."
        )

    common_keys = common_frame[
        [HIVE_COLUMN, TIMESTAMP_COLUMN]
    ]
    event_match = events.merge(
        common_keys,
        left_on=[HIVE_COLUMN, "event_start"],
        right_on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        how="left",
        indicator=True,
    )
    unmatched_events = event_match.loc[
        event_match["_merge"].ne("both")
    ]
    if not unmatched_events.empty:
        ids = unmatched_events[
            "harvest_event_id"
        ].astype(str).tolist()
        raise ValueError(
            "Reviewed events do not match common sensor rows: "
            f"{ids}"
        )

    manifest_columns = [
        HIVE_COLUMN,
        TIMESTAMP_COLUMN,
        "split",
    ]
    if "is_boundary_gap" in manifest.columns:
        manifest_columns.append("is_boundary_gap")

    base = common_frame.drop(
        columns=[
            "split",
            "is_boundary_gap",
            target_column,
            event_indicator_column,
            "harvest_event_id",
            "is_post_event_recovery_gap",
        ],
        errors="ignore",
    ).merge(
        manifest[manifest_columns],
        on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        how="left",
        validate="one_to_one",
    )

    if base["split"].isna().any():
        raise ValueError(
            "Some common rows did not match the split manifest."
        )

    if "is_boundary_gap" not in base.columns:
        base["is_boundary_gap"] = False
    base["is_boundary_gap"] = (
        base["is_boundary_gap"]
        .fillna(False)
        .astype(bool)
    )

    prepared = make_future_reviewed_event_target(
        base,
        events,
        horizon_hours=horizon_hours,
        output_column=target_column,
    )
    prepared = add_reviewed_event_columns(
        prepared,
        events,
        indicator_column=event_indicator_column,
    )
    prepared = add_post_event_recovery_gap(
        prepared,
        events,
        recovery_hours=post_event_recovery_hours,
    )

    unavailable_target_rows = int(
        prepared[target_column].isna().sum()
    )
    boundary_gap_rows = int(
        prepared["is_boundary_gap"].sum()
    )
    recovery_gap_rows = int(
        prepared[
            "is_post_event_recovery_gap"
        ].sum()
    )

    modelling = prepared.loc[
        prepared[target_column].notna()
        & ~prepared["is_boundary_gap"]
        & ~prepared["is_post_event_recovery_gap"]
    ].copy()

    modelling[target_column] = modelling[
        target_column
    ].astype("int8")

    balance = (
        modelling.groupby(
            ["split", target_column],
            observed=True,
        )
        .size()
        .rename("rows")
        .reset_index()
    )

    event_counts = (
        events.groupby("split", observed=True)
        .size()
        .to_dict()
        if "split" in events.columns
        else {}
    )

    audit: dict[str, Any] = {
        "source_rows": len(common_frame),
        "reviewed_event_count": len(events),
        "reviewed_positive_hives": int(
            events[HIVE_COLUMN].nunique()
        ),
        "prediction_horizon_hours": horizon_hours,
        "post_event_recovery_hours": (
            post_event_recovery_hours
        ),
        "rows_with_unavailable_future_target": (
            unavailable_target_rows
        ),
        "boundary_gap_rows_removed": boundary_gap_rows,
        "post_event_recovery_rows_removed": (
            recovery_gap_rows
        ),
        "final_modelling_rows": len(modelling),
        "target_positive_rows": int(
            modelling[target_column].sum()
        ),
        "target_negative_rows": int(
            len(modelling)
            - modelling[target_column].sum()
        ),
        "target_positive_rate": float(
            modelling[target_column].mean()
        ),
        "reviewed_events_by_split": {
            str(key): int(value)
            for key, value in event_counts.items()
        },
        "warning": (
            "Reviewed events remain probable pseudo-harvest events, "
            "not beekeeper-confirmed ground truth. Report independent "
            "event counts in every split."
        ),
    }

    modelling = modelling.sort_values(
        [HIVE_COLUMN, TIMESTAMP_COLUMN]
    ).reset_index(drop=True)

    return modelling, audit, balance


def run_reviewed_dataset_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    common_path = _resolve_path(
        root,
        config["dataset"]["clean_data_path"],
    )
    manifest_path = _resolve_path(
        root,
        config["dataset"]["split_manifest_path"],
    )
    event_path = _resolve_path(
        root,
        config["reviewed"]["event_table_path"],
    )
    output_path = _resolve_path(
        root,
        config["reviewed"]["model_dataset_path"],
    )
    report_directory = _resolve_path(
        root,
        config["reviewed"]["report_directory"],
    )

    common = pd.read_parquet(common_path)
    manifest = pd.read_parquet(manifest_path)
    events = pd.read_parquet(event_path)

    modelling, audit, balance = (
        build_reviewed_harvest_dataset(
            common,
            manifest,
            events,
            horizon_hours=int(
                config["target"]["horizon_hours"]
            ),
            target_column=config[
                "reviewed_target"
            ]["output_column"],
            event_indicator_column=config[
                "reviewed_target"
            ]["event_start_indicator_column"],
            post_event_recovery_hours=int(
                config["reviewed_target"][
                    "post_event_recovery_hours"
                ]
            ),
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    modelling.to_parquet(
        output_path,
        index=False,
    )
    balance.to_csv(
        report_directory
        / "target_balance_by_split.csv",
        index=False,
    )
    (
        report_directory / "target_audit.json"
    ).write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    return {
        **audit,
        "model_dataset_path": str(output_path),
        "report_directory": str(report_directory),
    }
