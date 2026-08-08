from __future__ import annotations

import argparse
import json
from pathlib import Path

from multivari.modules.harvesting.label_audit import (
    run_label_alignment_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Audit temporal alignment and sensor quality of generated harvest markers.")
    )
    parser.add_argument(
        "--config",
        default="config/harvesting.yaml",
        help=("Path relative to backend, or an absolute YAML path."),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    backend_root = Path(__file__).resolve().parents[1]

    summary = run_label_alignment_audit(
        backend_root=backend_root,
        config_path=arguments.config,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
