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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Create event-aware chronological validation folds for the harvesting module.")
    )

    parser.add_argument(
        "--config",
        default="config/harvesting.yaml",
        help=("Configuration path relative to backend, or an absolute path."),
    )

    return parser.parse_args()


def resolve_path(
    backend_root: Path,
    configured_path: str,
) -> Path:
    path = Path(configured_path)

    if path.is_absolute():
        return path

    return backend_root / path


def main() -> None:
    args = parse_args()

    backend_root = Path(__file__).resolve().parents[1]

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = backend_root / config_path

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    event_table_path = resolve_path(
        backend_root,
        config["output"]["event_table_path"],
    )

    output_path = resolve_path(
        backend_root,
        config["output"]["cv_folds_path"],
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    events = pd.read_parquet(event_table_path)

    cross_validation = config["cross_validation"]

    folds = create_event_aware_folds(
        events,
        minimum_training_events=int(cross_validation["minimum_training_events"]),
        validation_events_per_fold=int(cross_validation["validation_events_per_fold"]),
        prediction_horizon_hours=int(config["target"]["horizon_hours"]),
        purge_hours=int(cross_validation["purge_hours"]),
    )

    fold_frame = folds_to_frame(folds)

    fold_frame.to_csv(
        output_path,
        index=False,
    )

    summary = {
        "event_table_path": str(event_table_path),
        "output_path": str(output_path),
        "fold_count": len(fold_frame),
        "minimum_training_events": int(cross_validation["minimum_training_events"]),
        "validation_events_per_fold": int(cross_validation["validation_events_per_fold"]),
        "prediction_horizon_hours": int(config["target"]["horizon_hours"]),
        "purge_hours": int(cross_validation["purge_hours"]),
        "folds": fold_frame.to_dict(orient="records"),
    }

    print(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
