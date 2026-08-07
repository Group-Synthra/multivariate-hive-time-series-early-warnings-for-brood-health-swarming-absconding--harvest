from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = BACKEND_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from multivari.modules.absconding import (
    AbscondingSettings,
    run_absconding_data_pipeline,
    run_absconding_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the leakage-safe Absconding early-warning module."
    )
    parser.add_argument(
        "--config",
        default=str(BACKEND_ROOT / "config" / "absconding.yaml"),
        help="Absconding YAML configuration path.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Optional separate Absconding CSV/XLSX. When supplied, the module-specific data "
            "pipeline runs before model training."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model keys to run instead of the configured candidate list.",
    )
    args = parser.parse_args()

    settings = AbscondingSettings.from_yaml(args.config)
    clean_path = BACKEND_ROOT / settings.data_clean_path
    manifest_path = BACKEND_ROOT / settings.data_manifest_path
    source_path = Path(args.input or settings.data_input_path)
    if not source_path.is_absolute():
        source_path = BACKEND_ROOT / source_path

    if args.input or not clean_path.is_file() or not manifest_path.is_file():
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Separate Absconding dataset not found: {source_path}. "
                "Copy hive_data_with_features.csv to data/raw/absconding or pass --input."
            )
        profile = run_absconding_data_pipeline(
            input_path=source_path,
            backend_root=BACKEND_ROOT,
            config_path=args.config,
        )
        print(
            "Prepared separate Absconding data: "
            f"{profile.rows_clean:,} rows, {profile.merged_event_episodes} episodes."
        )

    dashboard = run_absconding_pipeline(
        backend_root=BACKEND_ROOT,
        config_path=args.config,
        model_candidates=tuple(args.models) if args.models else None,
    )
    summary = dashboard["summary"]
    metrics = dashboard["model_training"]["test_metrics"]
    event_metrics = dashboard["model_training"]["test_event_metrics"]
    print("Absconding pipeline completed.")
    print(f"Selected model: {summary['selected_model_name']}")
    print(f"Test PR-AUC: {metrics['pr_auc']}")
    print(f"Test recall: {metrics['recall']}")
    print(f"Test event recall: {event_metrics['event_recall']}")
    print("Dashboard JSON: artifacts/reports/absconding/absconding_dashboard.json")


if __name__ == "__main__":
    main()
