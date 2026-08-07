from __future__ import annotations

import argparse
import json
from pathlib import Path

from multivari.modules.harvesting.dataset import (
    build_harvest_modelling_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build the harvesting 72-hour modelling dataset.")
    )

    parser.add_argument(
        "--config",
        default="config/harvesting.yaml",
        help=("Path relative to backend, or an absolute YAML path."),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    backend_root = Path(__file__).resolve().parents[1]

    audit = build_harvest_modelling_dataset(
        backend_root=backend_root,
        config_path=args.config,
    )

    print(
        json.dumps(
            audit,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
