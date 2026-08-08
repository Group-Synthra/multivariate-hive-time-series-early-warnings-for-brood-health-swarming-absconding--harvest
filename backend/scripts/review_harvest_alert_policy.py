from __future__ import annotations

import argparse
import json
from pathlib import Path

from multivari.modules.harvesting.alert_policy_research_gate import (
    run_alert_policy_research_gate_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Apply the research gate to the harvest alert-policy sweep.")
    )
    parser.add_argument(
        "--config",
        default="config/harvesting.yaml",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    backend_root = Path(__file__).resolve().parents[1]
    result = run_alert_policy_research_gate_from_config(
        backend_root=backend_root,
        config_path=arguments.config,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
