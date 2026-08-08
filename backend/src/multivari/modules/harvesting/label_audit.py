from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

HIVE_COLUMN = "hive_id"
TIMESTAMP_COLUMN = "timestamp"
WEIGHT_COLUMN = "weight_kg"
CO2_COLUMN = "co2_ppm"


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


def _robust_mad(series: pd.Series) -> float:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0

    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _find_candidate_drop_onset(
    hive_frame: pd.DataFrame,
    *,
    marker_time: pd.Timestamp,
    lookback_hours: int,
    persistence_hours: int,
    minimum_drop_kg: float,
    mad_multiplier: float,
    minimum_persistent_drop_kg: float,
) -> dict[str, Any]:
    start = marker_time - pd.Timedelta(hours=lookback_hours)
    end = marker_time + pd.Timedelta(hours=persistence_hours)

    window = hive_frame.loc[
        hive_frame[TIMESTAMP_COLUMN].between(
            start,
            end,
            inclusive="both",
        )
    ].copy()

    if window.empty:
        return {
            "candidate_drop_onset": pd.NaT,
            "candidate_drop_1h_kg": np.nan,
            "persistent_drop_kg": np.nan,
            "drop_threshold_kg": np.nan,
        }

    window = window.sort_values(TIMESTAMP_COLUMN)
    window["weight_change_1h"] = window[WEIGHT_COLUMN].diff()

    baseline_changes = window.loc[
        window[TIMESTAMP_COLUMN] < marker_time,
        "weight_change_1h",
    ]
    mad = _robust_mad(baseline_changes)
    threshold = max(
        float(minimum_drop_kg),
        float(mad_multiplier) * mad,
    )

    candidate_rows = window.loc[
        window[TIMESTAMP_COLUMN].between(
            start,
            marker_time,
            inclusive="both",
        )
        & window["weight_change_1h"].le(-threshold)
    ]

    for candidate in candidate_rows.itertuples(index=False):
        candidate_time = pd.Timestamp(getattr(candidate, TIMESTAMP_COLUMN))
        candidate_weight = float(getattr(candidate, WEIGHT_COLUMN))
        change_1h = float(candidate.weight_change_1h)

        previous_rows = window.loc[window[TIMESTAMP_COLUMN] < candidate_time]
        if previous_rows.empty:
            continue

        pre_weight = float(previous_rows.iloc[-1][WEIGHT_COLUMN])

        persistence_end = candidate_time + pd.Timedelta(hours=persistence_hours)
        future_rows = window.loc[
            window[TIMESTAMP_COLUMN].between(
                candidate_time,
                persistence_end,
                inclusive="both",
            ),
            WEIGHT_COLUMN,
        ].dropna()

        if future_rows.empty:
            continue

        future_weight = float(future_rows.median())
        persistent_drop = pre_weight - future_weight

        if persistent_drop >= minimum_persistent_drop_kg:
            return {
                "candidate_drop_onset": candidate_time,
                "candidate_drop_1h_kg": abs(change_1h),
                "persistent_drop_kg": persistent_drop,
                "drop_threshold_kg": threshold,
                "candidate_weight_kg": candidate_weight,
                "pre_drop_weight_kg": pre_weight,
            }

    return {
        "candidate_drop_onset": pd.NaT,
        "candidate_drop_1h_kg": np.nan,
        "persistent_drop_kg": np.nan,
        "drop_threshold_kg": threshold,
        "candidate_weight_kg": np.nan,
        "pre_drop_weight_kg": np.nan,
    }


def audit_harvest_event_alignment(
    common: pd.DataFrame,
    events: pd.DataFrame,
    *,
    event_id_column: str = "harvest_event_id",
    event_start_column: str = "event_start",
    lookback_hours: int = 72,
    persistence_hours: int = 6,
    minimum_drop_kg: float = 2.0,
    mad_multiplier: float = 6.0,
    minimum_persistent_drop_kg: float = 1.5,
    aligned_tolerance_hours: int = 2,
    co2_flatline_std_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Audit whether each harvest marker is aligned with the onset of a
    sustained weight reduction.

    The result is an audit table. It does not silently change labels.
    """
    _require_columns(
        common,
        {
            HIVE_COLUMN,
            TIMESTAMP_COLUMN,
            WEIGHT_COLUMN,
            CO2_COLUMN,
        },
        frame_name="Common cleaned dataset",
    )
    _require_columns(
        events,
        {
            HIVE_COLUMN,
            event_id_column,
            event_start_column,
        },
        frame_name="Harvest event table",
    )

    common_frame = common.copy()
    common_frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        common_frame[TIMESTAMP_COLUMN],
        errors="raise",
    )
    common_frame = common_frame.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN])

    event_frame = events.copy()
    event_frame[event_start_column] = pd.to_datetime(
        event_frame[event_start_column],
        errors="raise",
    )

    rows: list[dict[str, Any]] = []

    for event in event_frame.itertuples(index=False):
        values = event._asdict()
        hive_id = values[HIVE_COLUMN]
        event_id = values[event_id_column]
        marker_time = pd.Timestamp(values[event_start_column])

        hive_frame = common_frame.loc[common_frame[HIVE_COLUMN].eq(hive_id)]

        detection = _find_candidate_drop_onset(
            hive_frame,
            marker_time=marker_time,
            lookback_hours=lookback_hours,
            persistence_hours=persistence_hours,
            minimum_drop_kg=minimum_drop_kg,
            mad_multiplier=mad_multiplier,
            minimum_persistent_drop_kg=minimum_persistent_drop_kg,
        )

        onset = detection["candidate_drop_onset"]

        if pd.isna(onset):
            marker_delay_hours = np.nan
            alignment_status = "no_clear_sustained_drop"
        else:
            marker_delay_hours = (marker_time - pd.Timestamp(onset)).total_seconds() / 3600

            if abs(marker_delay_hours) <= aligned_tolerance_hours:
                alignment_status = "aligned"
            elif marker_delay_hours > aligned_tolerance_hours:
                alignment_status = "marker_delayed"
            else:
                alignment_status = "marker_before_detected_drop"

        pre_window = hive_frame.loc[
            hive_frame[TIMESTAMP_COLUMN].between(
                marker_time - pd.Timedelta(hours=72),
                marker_time - pd.Timedelta(hours=1),
                inclusive="both",
            )
        ]

        co2_std_pre72h = float(pre_window[CO2_COLUMN].std())
        co2_unique_pre72h = int(pre_window[CO2_COLUMN].nunique(dropna=True))
        co2_flatline_pre72h = bool(
            co2_unique_pre72h <= 1
            or (not np.isnan(co2_std_pre72h) and co2_std_pre72h <= co2_flatline_std_threshold)
        )

        weight_std_pre24h = float(pre_window.tail(24)[WEIGHT_COLUMN].std())

        row: dict[str, Any] = {
            HIVE_COLUMN: hive_id,
            event_id_column: event_id,
            "marker_event_start": marker_time,
            **detection,
            "marker_delay_hours": marker_delay_hours,
            "alignment_status": alignment_status,
            "weight_std_pre24h": weight_std_pre24h,
            "co2_std_pre72h": co2_std_pre72h,
            "co2_unique_values_pre72h": co2_unique_pre72h,
            "co2_flatline_pre72h": int(co2_flatline_pre72h),
            "manual_event_type": "",
            "manual_include_for_training": "",
            "manual_reviewed_event_start": "",
            "manual_reviewer_notes": "",
        }

        for optional_column in (
            "split",
            "positive_rows",
            "event_duration_hours",
        ):
            if optional_column in values:
                row[optional_column] = values[optional_column]

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_alignment_audit(
    audit: pd.DataFrame,
) -> dict[str, Any]:
    status_counts = audit["alignment_status"].value_counts(dropna=False).to_dict()

    delayed = audit.loc[
        audit["alignment_status"].eq("marker_delayed"),
        "marker_delay_hours",
    ].dropna()

    return {
        "audited_events": len(audit),
        "alignment_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "median_marker_delay_hours_for_delayed_events": (
            float(delayed.median()) if not delayed.empty else None
        ),
        "co2_flatline_events_pre72h": int(audit["co2_flatline_pre72h"].sum()),
        "manual_review_required": True,
        "decision_rule": (
            "Do not overwrite event times automatically. Review each "
            "individual event plot, classify the event, and enter a "
            "reviewed event start before rebuilding the prediction target."
        ),
    }


def run_label_alignment_audit(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    configuration_path = Path(config_path)
    if not configuration_path.is_absolute():
        configuration_path = root / configuration_path

    config = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))

    common_path = _resolve_path(
        root,
        config["dataset"]["clean_data_path"],
    )
    event_path = _resolve_path(
        root,
        config["output"]["event_table_path"],
    )

    audit_config = config["label_audit"]
    output_directory = _resolve_path(
        root,
        audit_config["output_directory"],
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    common = pd.read_parquet(common_path)
    events = pd.read_parquet(event_path)

    audit = audit_harvest_event_alignment(
        common,
        events,
        event_id_column=config["event"]["event_id_column"],
        event_start_column="event_start",
        lookback_hours=int(audit_config["lookback_hours"]),
        persistence_hours=int(audit_config["persistence_hours"]),
        minimum_drop_kg=float(audit_config["minimum_drop_kg"]),
        mad_multiplier=float(audit_config["mad_multiplier"]),
        minimum_persistent_drop_kg=float(audit_config["minimum_persistent_drop_kg"]),
        aligned_tolerance_hours=int(audit_config["aligned_tolerance_hours"]),
        co2_flatline_std_threshold=float(audit_config["co2_flatline_std_threshold"]),
    )

    summary = summarize_alignment_audit(audit)

    audit.to_csv(
        output_directory / "event_label_alignment_audit.csv",
        index=False,
    )
    (output_directory / "event_label_alignment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary
