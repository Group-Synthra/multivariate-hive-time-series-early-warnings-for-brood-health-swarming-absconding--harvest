from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from multivari.common.io import read_table, write_parquet
from multivari.common.schema import HIVE_COLUMN
from multivari.common.splitting import join_split_manifest
from multivari.common.targets import (
    remove_target_leakage_boundaries,
)

from .events import (
    add_harvest_event_identifiers,
    build_harvest_event_table,
)
from .targets import add_future_harvest_target


def _resolve_path(
    root: Path,
    configured_path: str,
) -> Path:
    path = Path(configured_path)

    if path.is_absolute():
        return path

    return root / path


def _target_balance_by_split(
    df: pd.DataFrame,
    *,
    target_column: str,
    event_start_column: str,
) -> pd.DataFrame:
    balance = (
        df.groupby(
            "split",
            observed=True,
        )
        .agg(
            rows=(target_column, "size"),
            target_positives=(target_column, "sum"),
            event_starts=(event_start_column, "sum"),
        )
        .reset_index()
    )

    balance["target_positives"] = balance["target_positives"].astype("int64")

    balance["event_starts"] = balance["event_starts"].astype("int64")

    balance["target_negatives"] = balance["rows"] - balance["target_positives"]

    balance["positive_rate"] = balance["target_positives"] / balance["rows"]

    return balance


def build_harvest_modelling_dataset(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Build the leakage-controlled 72-hour harvesting dataset."""
    root = Path(backend_root).resolve()

    config_file = Path(config_path)

    if not config_file.is_absolute():
        config_file = root / config_file

    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))

    clean_path = _resolve_path(
        root,
        config["dataset"]["clean_data_path"],
    )

    manifest_path = _resolve_path(
        root,
        config["dataset"]["split_manifest_path"],
    )

    event_output_path = _resolve_path(
        root,
        config["output"]["event_table_path"],
    )

    model_output_path = _resolve_path(
        root,
        config["output"]["model_dataset_path"],
    )

    report_directory = _resolve_path(
        root,
        config["output"]["report_directory"],
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    event_config = config["event"]
    target_config = config["target"]

    source_column = event_config["source_column"]
    event_start_column = event_config["event_start_column"]
    event_id_column = event_config["event_id_column"]

    target_column = target_config["output_column"]
    horizon_hours = int(target_config["horizon_hours"])

    common = read_table(clean_path)
    manifest = read_table(manifest_path)

    prepared = add_harvest_event_identifiers(
        common,
        source_column=source_column,
        event_start_column=event_start_column,
        event_id_column=event_id_column,
    )

    prepared = add_future_harvest_target(
        prepared,
        event_start_column=event_start_column,
        horizon_hours=horizon_hours,
        output_column=target_column,
    )

    prepared = join_split_manifest(
        prepared,
        manifest,
    )

    if prepared["split"].isna().any():
        raise ValueError("Some rows did not match the common split manifest.")

    if prepared["is_boundary_gap"].isna().any():
        raise ValueError("Some rows are missing the boundary-gap flag.")

    event_table = build_harvest_event_table(
        prepared,
        source_column=source_column,
        event_start_column=event_start_column,
        event_id_column=event_id_column,
    )

    source_positive_rows = int(prepared[source_column].sum())

    consolidated_events = int(prepared[event_start_column].sum())

    positive_hives = int(
        prepared.loc[
            prepared[event_start_column].eq(1),
            HIVE_COLUMN,
        ].nunique()
    )

    unavailable_future_rows = int(prepared[target_column].isna().sum())

    boundary_gap_rows = int(prepared["is_boundary_gap"].sum())

    # Remove the rows reserved around train/validation/test
    # boundaries.
    model_data = remove_target_leakage_boundaries(prepared)

    # The last 72 hours of each hive have no complete future
    # observation window, so their target remains unknown.
    model_data = model_data.dropna(subset=[target_column]).copy()

    model_data[target_column] = model_data[target_column].astype("int8")

    model_data = model_data.sort_values([HIVE_COLUMN, "timestamp"]).reset_index(drop=True)

    target_balance = _target_balance_by_split(
        model_data,
        target_column=target_column,
        event_start_column=event_start_column,
    )

    eligible_events = event_table.loc[~event_table["is_boundary_gap"].fillna(False)]

    events_by_split = {
        str(split): int(count)
        for split, count in (
            eligible_events.groupby(
                "split",
                observed=True,
            )
            .size()
            .items()
        )
    }

    audit = {
        "source_data_path": str(clean_path),
        "split_manifest_path": str(manifest_path),
        "prediction_horizon_hours": horizon_hours,
        "source_positive_rows": source_positive_rows,
        "consolidated_event_count": consolidated_events,
        "positive_hives": positive_hives,
        "rows_with_unavailable_future_target": (unavailable_future_rows),
        "boundary_gap_rows_removed": boundary_gap_rows,
        "final_modelling_rows": len(model_data),
        "target_positive_rows": int(model_data[target_column].sum()),
        "target_negative_rows": int(len(model_data) - model_data[target_column].sum()),
        "event_starts_by_split": events_by_split,
        "warning": (
            "Harvest markers are generated labels. "
            "Evaluate results as a prototype and report "
            "the number of independent events in every split."
        ),
    }

    write_parquet(
        event_table,
        event_output_path,
    )

    write_parquet(
        model_data,
        model_output_path,
    )

    event_table.to_csv(
        report_directory / "harvest_events.csv",
        index=False,
    )

    target_balance.to_csv(
        report_directory / "target_balance_by_split.csv",
        index=False,
    )

    (report_directory / "target_audit.json").write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    return audit
