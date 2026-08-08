from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Merge provisional HUI regression configuration into config/harvesting.yaml.")
    )
    parser.add_argument(
        "--config",
        default="config/harvesting.yaml",
    )
    parser.add_argument(
        "--snippet",
        default=("config/harvesting_provisional_hui_regression_section.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    backend_root = Path(__file__).resolve().parents[1]
    config_path = backend_root / arguments.config
    snippet_path = backend_root / arguments.snippet

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    snippet = yaml.safe_load(snippet_path.read_text(encoding="utf-8"))

    if not isinstance(config, dict) or not isinstance(snippet, dict):
        raise TypeError("Both YAML files must contain top-level mappings.")

    for key, value in snippet.items():
        config[key] = value

    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    print("Merged sections:", list(snippet))
    print("Updated:", config_path)


if __name__ == "__main__":
    main()
