from __future__ import annotations

import argparse
import json
from pathlib import Path

from multivari.modules.harvesting.eda import run_harvest_research_eda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run research-grade, event-aware harvesting EDA."
    )
    parser.add_argument(
        "--config",
        default="config/harvesting.yaml",
        help="Path relative to backend, or an absolute YAML path.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    backend_root = Path(__file__).resolve().parents[1]
    summary = run_harvest_research_eda(
        backend_root=backend_root,
        config_path=arguments.config,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
