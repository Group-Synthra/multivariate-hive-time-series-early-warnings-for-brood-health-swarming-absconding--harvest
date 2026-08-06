from __future__ import annotations

import argparse
import json
from pathlib import Path

from multivari.modules.harvesting.reviewed_dataset import (
    run_reviewed_dataset_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the 72-hour harvest dataset from "
            "manually reviewed probable events."
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

    summary = run_reviewed_dataset_from_config(
        backend_root=backend_root,
        config_path=arguments.config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
