from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

HIVE_COLUMN = "hive_id"
TIMESTAMP_COLUMN = "timestamp"
ALLOWED_EVENT_TYPES = {
    "probable_harvest",
    "equipment_change",
    "sensor_error",
    "unclear",
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


def create_manual_review_template(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """Create a review sheet without silently accepting suggestions."""
    _require_columns(
        audit,
        {
            HIVE_COLUMN,
            "harvest_event_id",
            "marker_event_start",
            "candidate_drop_onset",
            "marker_delay_hours",
            "alignment_status",
            "persistent_drop_kg",
            "co2_flatline_pre72h",
        },
        frame_name="Label-alignment audit",
    )

    review = audit.copy()

    review["suggested_event_type"] = "unclear"
    review["suggested_include_for_training"] = 0
    review["suggested_reviewed_event_start"] = ""

    aligned = review["alignment_status"].eq("aligned")
    delayed = review["alignment_status"].eq("marker_delayed")

    review.loc[
        aligned,
        "suggested_event_type",
    ] = "probable_harvest"
    review.loc[
        aligned,
        "suggested_include_for_training",
    ] = 1
    review.loc[
        aligned,
        "suggested_reviewed_event_start",
    ] = review.loc[aligned, "marker_event_start"].astype(str)

    review.loc[
        delayed,
        "suggested_event_type",
    ] = "probable_harvest"
    review.loc[
        delayed,
        "suggested_include_for_training",
    ] = 1
    review.loc[
        delayed,
        "suggested_reviewed_event_start",
    ] = review.loc[delayed, "candidate_drop_onset"].astype(str)

    review["manual_event_type"] = ""
    review["manual_include_for_training"] = ""
    review["manual_reviewed_event_start"] = ""
    review["manual_reviewer_notes"] = ""
    review["manual_review_complete"] = 0

    preferred_columns = [
        HIVE_COLUMN,
        "harvest_event_id",
        "split",
        "marker_event_start",
        "candidate_drop_onset",
        "marker_delay_hours",
        "alignment_status",
        "candidate_drop_1h_kg",
        "persistent_drop_kg",
        "weight_std_pre24h",
        "co2_std_pre72h",
        "co2_unique_values_pre72h",
        "co2_flatline_pre72h",
        "suggested_event_type",
        "suggested_include_for_training",
        "suggested_reviewed_event_start",
        "manual_event_type",
        "manual_include_for_training",
        "manual_reviewed_event_start",
        "manual_reviewer_notes",
        "manual_review_complete",
    ]

    existing = [column for column in preferred_columns if column in review.columns]
    remaining = [column for column in review.columns if column not in existing]

    return review[existing + remaining]


def validate_manual_review(
    review: pd.DataFrame,
) -> None:
    """Validate that every event received an explicit manual decision."""
    _require_columns(
        review,
        {
            HIVE_COLUMN,
            "harvest_event_id",
            "manual_event_type",
            "manual_include_for_training",
            "manual_reviewed_event_start",
            "manual_review_complete",
        },
        frame_name="Manual event review",
    )

    incomplete = review.loc[
        pd.to_numeric(
            review["manual_review_complete"],
            errors="coerce",
        )
        .fillna(0)
        .ne(1)
    ]
    if not incomplete.empty:
        ids = incomplete["harvest_event_id"].astype(str).tolist()
        raise ValueError(f"Manual review is incomplete for events: {ids}")

    event_types = review["manual_event_type"].astype("string").str.strip()
    invalid_types = sorted(set(event_types.dropna()).difference(ALLOWED_EVENT_TYPES))
    if invalid_types:
        raise ValueError(
            "Invalid manual_event_type values: "
            f"{invalid_types}. Allowed values are "
            f"{sorted(ALLOWED_EVENT_TYPES)}."
        )

    include_values = pd.to_numeric(
        review["manual_include_for_training"],
        errors="coerce",
    )
    invalid_include = ~include_values.isin([0, 1])
    if invalid_include.any():
        ids = (
            review.loc[
                invalid_include,
                "harvest_event_id",
            ]
            .astype(str)
            .tolist()
        )
        raise ValueError(f"manual_include_for_training must be 0 or 1 for: {ids}")

    included = include_values.eq(1)
    included_types = event_types.loc[included]
    invalid_included_types = included_types.ne("probable_harvest")
    if invalid_included_types.any():
        ids = (
            review.loc[
                included_types.index[invalid_included_types],
                "harvest_event_id",
            ]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"Only probable_harvest events may be included for training. Invalid events: {ids}"
        )

    reviewed_times = pd.to_datetime(
        review["manual_reviewed_event_start"],
        errors="coerce",
    )
    missing_times = included & reviewed_times.isna()
    if missing_times.any():
        ids = (
            review.loc[
                missing_times,
                "harvest_event_id",
            ]
            .astype(str)
            .tolist()
        )
        raise ValueError(f"Included events require a valid manual_reviewed_event_start: {ids}")


def build_reviewed_event_table(
    review: pd.DataFrame,
    common: pd.DataFrame,
    split_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the final reviewed event table.

    Reviewed timestamps must match an actual hourly record exactly.
    """
    validate_manual_review(review)

    _require_columns(
        common,
        {HIVE_COLUMN, TIMESTAMP_COLUMN},
        frame_name="Common cleaned dataset",
    )
    _require_columns(
        split_manifest,
        {HIVE_COLUMN, TIMESTAMP_COLUMN},
        frame_name="Split manifest",
    )

    include_values = pd.to_numeric(
        review["manual_include_for_training"],
        errors="raise",
    )
    included = review.loc[include_values.eq(1)].copy()

    included["event_start"] = pd.to_datetime(
        included["manual_reviewed_event_start"],
        errors="raise",
    )

    common_keys = common[[HIVE_COLUMN, TIMESTAMP_COLUMN]].copy()
    common_keys[TIMESTAMP_COLUMN] = pd.to_datetime(
        common_keys[TIMESTAMP_COLUMN],
        errors="raise",
    )
    common_keys = common_keys.drop_duplicates()

    unmatched = included.merge(
        common_keys,
        left_on=[HIVE_COLUMN, "event_start"],
        right_on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        how="left",
        indicator=True,
    )
    unmatched = unmatched.loc[unmatched["_merge"].ne("both")]
    if not unmatched.empty:
        ids = unmatched["harvest_event_id"].astype(str).tolist()
        raise ValueError(
            "Reviewed event timestamps must exactly match "
            f"common hourly records. Unmatched events: {ids}"
        )

    manifest = split_manifest.copy()
    manifest[TIMESTAMP_COLUMN] = pd.to_datetime(
        manifest[TIMESTAMP_COLUMN],
        errors="raise",
    )

    manifest_columns = [
        HIVE_COLUMN,
        TIMESTAMP_COLUMN,
    ]
    for optional in ("split", "is_boundary_gap"):
        if optional in manifest.columns:
            manifest_columns.append(optional)

    manifest = manifest[manifest_columns].drop_duplicates(subset=[HIVE_COLUMN, TIMESTAMP_COLUMN])

    reviewed = included.merge(
        manifest,
        left_on=[HIVE_COLUMN, "event_start"],
        right_on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        how="left",
        validate="many_to_one",
        suffixes=("", "_reviewed"),
    )

    if "split_reviewed" in reviewed.columns:
        reviewed["split"] = reviewed["split_reviewed"]
        reviewed = reviewed.drop(columns=["split_reviewed"])

    if "is_boundary_gap_reviewed" in reviewed.columns:
        reviewed["is_boundary_gap"] = reviewed["is_boundary_gap_reviewed"]
        reviewed = reviewed.drop(columns=["is_boundary_gap_reviewed"])

    reviewed = reviewed.drop(
        columns=[TIMESTAMP_COLUMN],
        errors="ignore",
    )

    reviewed = reviewed.sort_values([HIVE_COLUMN, "event_start"]).reset_index(drop=True)

    reviewed["source_harvest_event_id"] = reviewed["harvest_event_id"]
    reviewed["event_number"] = reviewed.groupby(HIVE_COLUMN).cumcount().add(1)
    reviewed["harvest_event_id"] = (
        reviewed[HIVE_COLUMN].astype(str)
        + "_harvest_"
        + reviewed["event_number"].astype(str).str.zfill(3)
    )

    duplicates = reviewed.duplicated(
        subset=[HIVE_COLUMN, "event_start"],
        keep=False,
    )
    if duplicates.any():
        rows = reviewed.loc[
            duplicates,
            [
                HIVE_COLUMN,
                "event_start",
                "source_harvest_event_id",
            ],
        ].to_dict(orient="records")
        raise ValueError(f"Duplicate reviewed event timestamps were found: {rows}")

    output_columns = [
        HIVE_COLUMN,
        "harvest_event_id",
        "source_harvest_event_id",
        "event_number",
        "event_start",
        "split",
        "is_boundary_gap",
        "alignment_status",
        "marker_event_start",
        "candidate_drop_onset",
        "marker_delay_hours",
        "persistent_drop_kg",
        "co2_flatline_pre72h",
        "manual_event_type",
        "manual_reviewer_notes",
    ]

    return reviewed[[column for column in output_columns if column in reviewed.columns]]


def summarize_review(
    review: pd.DataFrame,
    reviewed_events: pd.DataFrame,
) -> dict[str, Any]:
    event_types = review["manual_event_type"].astype("string").value_counts(dropna=False).to_dict()
    split_counts = (
        reviewed_events["split"].value_counts(dropna=False).to_dict()
        if "split" in reviewed_events.columns
        else {}
    )

    return {
        "audited_events": len(review),
        "included_probable_harvest_events": len(reviewed_events),
        "excluded_events": len(review) - len(reviewed_events),
        "manual_event_type_counts": {str(key): int(value) for key, value in event_types.items()},
        "included_events_by_split": {str(key): int(value) for key, value in split_counts.items()},
        "next_action": (
            "Rebuild the 72-hour future target from the reviewed "
            "event timestamps, then regenerate cross-validation folds."
        ),
    }


def prepare_review_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    review_config = config["review"]

    audit_path = _resolve_path(
        root,
        review_config["audit_path"],
    )
    output_path = _resolve_path(
        root,
        review_config["review_template_path"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(audit_path)
    template = create_manual_review_template(audit)
    template.to_csv(output_path, index=False)

    return {
        "audit_path": str(audit_path),
        "review_template_path": str(output_path),
        "events_requiring_review": len(template),
    }


def finalize_review_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    review_config = config["review"]

    review_path = _resolve_path(
        root,
        review_config["review_template_path"],
    )
    common_path = _resolve_path(
        root,
        config["dataset"]["clean_data_path"],
    )
    split_path = _resolve_path(
        root,
        config["dataset"]["split_manifest_path"],
    )
    csv_output = _resolve_path(
        root,
        review_config["reviewed_events_csv_path"],
    )
    parquet_output = _resolve_path(
        root,
        review_config["reviewed_events_parquet_path"],
    )
    summary_output = _resolve_path(
        root,
        review_config["review_summary_path"],
    )

    review = pd.read_csv(
        review_path,
        keep_default_na=False,
    )
    common = pd.read_parquet(common_path)
    split_manifest = pd.read_parquet(split_path)

    reviewed_events = build_reviewed_event_table(
        review,
        common,
        split_manifest,
    )
    summary = summarize_review(
        review,
        reviewed_events,
    )

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    parquet_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    reviewed_events.to_csv(csv_output, index=False)
    reviewed_events.to_parquet(parquet_output, index=False)
    summary_output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary
