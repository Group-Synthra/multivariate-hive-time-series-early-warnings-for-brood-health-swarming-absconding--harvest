from __future__ import annotations

import argparse
from pathlib import Path

from multivari.common.pipeline import run_common_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shared MULTIVARI data pipeline.")
    parser.add_argument("--input", required=True, help="Path to the immutable source XLSX/CSV")
    parser.add_argument(
        "--config",
        default="config/common.yaml",
        help="Path to the common YAML configuration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_common_pipeline(
        input_path=args.input,
        config_path=root / args.config,
        processed_directory=root / "data" / "processed",
        report_directory=root / "artifacts" / "reports",
        manifest_directory=root / "data" / "manifests",
    )


if __name__ == "__main__":
    main()
