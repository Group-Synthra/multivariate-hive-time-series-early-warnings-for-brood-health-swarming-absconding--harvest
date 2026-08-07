from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

HIVE_COLUMN = "hive_id"
SPLIT_COLUMN = "split"


def _resolve_path(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def create_grouped_positive_hive_folds(
    feature_dataset: pd.DataFrame,
    reviewed_events: pd.DataFrame,
    *,
    target_column: str,
    minimum_training_positive_hives: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_feature_columns = {
        HIVE_COLUMN,
        SPLIT_COLUMN,
        target_column,
    }
    missing_features = sorted(
        required_feature_columns.difference(
            feature_dataset.columns
        )
    )
    if missing_features:
        raise ValueError(
            "Feature dataset is missing required columns: "
            f"{missing_features}"
        )

    required_event_columns = {
        HIVE_COLUMN,
        SPLIT_COLUMN,
        "harvest_event_id",
    }
    missing_events = sorted(
        required_event_columns.difference(
            reviewed_events.columns
        )
    )
    if missing_events:
        raise ValueError(
            "Reviewed events are missing required columns: "
            f"{missing_events}"
        )

    training_events = reviewed_events.loc[
        reviewed_events[SPLIT_COLUMN].eq("train")
    ].copy()
    positive_hives = sorted(
        training_events[HIVE_COLUMN].unique().tolist()
    )

    if len(positive_hives) < (
        minimum_training_positive_hives + 1
    ):
        raise ValueError(
            "Not enough positive training hives for grouped "
            "validation."
        )

    training_rows = feature_dataset.loc[
        feature_dataset[SPLIT_COLUMN].eq("train")
    ].copy()

    records: list[dict[str, Any]] = []
    for fold_number, validation_hive in enumerate(
        positive_hives,
        start=1,
    ):
        training_positive_hives = [
            hive_id
            for hive_id in positive_hives
            if hive_id != validation_hive
        ]

        validation_rows = training_rows.loc[
            training_rows[HIVE_COLUMN].eq(validation_hive)
        ]
        fitting_rows = training_rows.loc[
            training_rows[HIVE_COLUMN].ne(validation_hive)
        ]

        validation_event_count = int(
            training_events[HIVE_COLUMN]
            .eq(validation_hive)
            .sum()
        )
        training_event_count = int(
            training_events[HIVE_COLUMN]
            .ne(validation_hive)
            .sum()
        )

        records.append(
            {
                "fold": fold_number,
                "validation_hive_id": validation_hive,
                "training_positive_hive_count": len(
                    training_positive_hives
                ),
                "training_positive_hives": "|".join(
                    training_positive_hives
                ),
                "training_event_count": training_event_count,
                "validation_event_count": (
                    validation_event_count
                ),
                "training_rows": len(fitting_rows),
                "validation_rows": len(validation_rows),
                "training_positive_rows": int(
                    fitting_rows[target_column].sum()
                ),
                "validation_positive_rows": int(
                    validation_rows[target_column].sum()
                ),
                "training_rule": (
                    "split == 'train' and hive_id != "
                    "validation_hive_id"
                ),
                "validation_rule": (
                    "split == 'train' and hive_id == "
                    "validation_hive_id"
                ),
            }
        )

    folds = pd.DataFrame(records)
    summary = {
        "status": "created",
        "fold_count": len(folds),
        "positive_training_hive_count": len(positive_hives),
        "positive_training_hives": positive_hives,
        "training_event_count": len(training_events),
        "minimum_training_positive_hives": (
            minimum_training_positive_hives
        ),
        "primary_evaluation": (
            "Official chronological validation with 2 events "
            "and a one-event final test case study."
        ),
        "secondary_evaluation": (
            "Leave-one-positive-hive-out sensitivity analysis "
            "inside the official training split."
        ),
        "warning": (
            "These grouped folds are secondary only. They do not "
            "replace chronological validation because many training "
            "events share the same date."
        ),
    }
    return folds, summary


def run_grouped_hive_validation_from_config(
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
    settings = config["grouped_hive_validation"]
    target_column = config["reviewed_target"]["output_column"]

    event_path = _resolve_path(
        root,
        settings["event_table_path"],
    )
    feature_path = _resolve_path(
        root,
        settings["feature_dataset_path"],
    )
    folds_path = _resolve_path(
        root,
        settings["folds_path"],
    )
    summary_path = _resolve_path(
        root,
        settings["summary_path"],
    )

    events = pd.read_parquet(event_path)
    features = pd.read_parquet(feature_path)

    folds, summary = create_grouped_positive_hive_folds(
        features,
        events,
        target_column=target_column,
        minimum_training_positive_hives=int(
            settings["minimum_training_positive_hives"]
        ),
    )

    folds_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds.to_csv(folds_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return {
        **summary,
        "folds_path": str(folds_path),
        "summary_path": str(summary_path),
    }
