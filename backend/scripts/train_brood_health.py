from __future__ import annotations

import argparse
import json

from multivari.modules.brood_health.eda import build_brood_eda
from multivari.modules.brood_health.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the future-window Brood Health Score regressor "
            "using unseen-hive validation and test partitions."
        )
    )
    parser.add_argument("--horizon-hours", type=int, default=6)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use smaller estimators and capped partitions for a quick development comparison.",
    )
    parser.add_argument(
        "--eda-only",
        action="store_true",
        help="Regenerate brood-health EDA JSON and report images without training.",
    )
    args = parser.parse_args()

    if args.eda_only:
        payload = build_brood_eda(save_cache=True)
        print(
            json.dumps(
                {
                    "records": payload["meta"]["records"],
                    "hives": payload["meta"]["hives"],
                    "images": len(payload["generated_images"]),
                },
                indent=2,
            )
        )
        return

    def progress(event: str, payload: dict) -> None:
        progress_value = payload.get("progress", 0)
        print(f"[{progress_value:>3}%] {event}: {payload.get('message', '')}", flush=True)

    summary = run_training(
        horizon_hours=args.horizon_hours,
        fast_mode=args.fast,
        progress_callback=progress,
    )
    metrics = summary["best_metrics"]
    interpretation = summary["accuracy_interpretation"]
    print(
        json.dumps(
            {
                "best_model": summary["best_model"],
                "test_mae_score_points": metrics["test_mae"],
                "test_rmse_score_points": metrics["test_rmse"],
                "test_r2": metrics["test_r2"],
                "overall_level_accuracy_percent": interpretation["overall_level_accuracy_percent"],
                "transition_level_accuracy_percent": interpretation["primary_early_warning_accuracy_percent"],
                "critical_recall": metrics["critical_recall"],
                "persistence_transition_accuracy_percent": interpretation[
                    "persistence_transition_level_accuracy_percent"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
