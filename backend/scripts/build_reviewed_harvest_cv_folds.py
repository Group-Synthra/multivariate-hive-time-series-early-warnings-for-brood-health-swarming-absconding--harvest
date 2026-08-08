from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from multivari.modules.harvesting.splitting import (
    create_event_aware_folds,
    folds_to_frame,
)


def _resolve_path(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build purged chronological folds from reviewed probable harvest events.")
    )
    parser.add_argument(
        "--config",
        default="config/harvesting.yaml",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    backend_root = Path(__file__).resolve().parents[1]
    config_path = Path(arguments.config)
    if not config_path.is_absolute():
        config_path = backend_root / config_path

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    event_path = _resolve_path(
        backend_root,
        config["reviewed"]["event_table_path"],
    )
    folds_path = _resolve_path(
        backend_root,
        config["reviewed"]["cv_folds_path"],
    )
    summary_path = _resolve_path(
        backend_root,
        config["reviewed"]["cv_summary_path"],
    )

    events = pd.read_parquet(event_path)
    usable_events = events.copy()

    if "is_boundary_gap" in usable_events.columns:
        usable_events = usable_events.loc[
            ~usable_events["is_boundary_gap"].fillna(False).astype(bool)
        ].copy()

    counts_by_split = (
        usable_events["split"].value_counts(dropna=False).to_dict()
        if "split" in usable_events.columns
        else {}
    )

    settings = config["reviewed_cross_validation"]

    folds_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        folds = create_event_aware_folds(
            usable_events,
            minimum_training_events=int(settings["minimum_training_events"]),
            validation_events_per_fold=int(settings["validation_events_per_fold"]),
            prediction_horizon_hours=int(config["target"]["horizon_hours"]),
            purge_hours=int(settings["purge_hours"]),
        )
        fold_frame = folds_to_frame(folds)
        status = "created"
        message = "Reviewed event-aware folds were created."
    except ValueError as error:
        fold_frame = pd.DataFrame(
            columns=[
                "fold",
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "training_events",
                "validation_events",
            ]
        )
        status = "insufficient_events"
        message = str(error)

    fold_frame.to_csv(
        folds_path,
        index=False,
    )

    summary = {
        "status": status,
        "message": message,
        "reviewed_event_count": len(events),
        "usable_event_count": len(usable_events),
        "events_by_split": {str(key): int(value) for key, value in counts_by_split.items()},
        "fold_count": len(fold_frame),
        "minimum_training_events": int(settings["minimum_training_events"]),
        "validation_events_per_fold": int(settings["validation_events_per_fold"]),
        "prediction_horizon_hours": int(config["target"]["horizon_hours"]),
        "purge_hours": int(settings["purge_hours"]),
        "folds_path": str(folds_path),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
