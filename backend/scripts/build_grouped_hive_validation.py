from __future__ import annotations

import argparse
import json
from pathlib import Path

from multivari.modules.harvesting.grouped_hive_validation import (
    run_grouped_hive_validation_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create leave-one-positive-hive-out sensitivity folds "
            "inside the official training split."
        )
    )
    parser.add_argument(
        "--config",
        default="config/harvesting.yaml",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    backend_root = Path(__file__).resolve().parents[1]

    summary = run_grouped_hive_validation_from_config(
        backend_root=backend_root,
        config_path=arguments.config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
