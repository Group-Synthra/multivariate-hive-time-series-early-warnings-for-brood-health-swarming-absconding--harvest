from __future__ import annotations

import argparse
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = BACKEND_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from multivari.modules.absconding import run_absconding_pipeline


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
        "--models",
        nargs="+",
        default=None,
        help="Optional model keys to run instead of the configured candidate list.",
    )
    args = parser.parse_args()

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
    print(
        "Dashboard JSON: "
        "artifacts/reports/absconding/absconding_dashboard.json"
    )


if __name__ == "__main__":
    main()
