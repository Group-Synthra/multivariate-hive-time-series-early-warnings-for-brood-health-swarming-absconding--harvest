from __future__ import annotations

import argparse
import json

from multivari.modules.brood_health.eda import build_brood_eda
from multivari.modules.brood_health.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the Brood Health v6 multi-horizon regressor. "
            "Primary output: exact score at +6 hours. Secondary output: minimum "
            "predicted score inside the 1–6 hour trajectory."
        )
    )
    parser.add_argument("--horizon-hours", type=int, default=6)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use smaller estimators and capped partitions for development.",
    )
    parser.add_argument(
        "--eda-only",
        action="store_true",
        help="Regenerate brood-health EDA without training.",
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
        value = payload.get("progress", 0)
        print(
            f"[{value:>3}%] {event}: {payload.get('message', '')}",
            flush=True,
        )

    summary = run_training(
        horizon_hours=args.horizon_hours,
        fast_mode=args.fast,
        progress_callback=progress,
    )
    metrics = summary["best_metrics"]
    exact = metrics["exact_horizon"]
    transition = metrics["transition"]
    forecast_indicators = metrics.get("forecast_indicators", {})
    print(
        json.dumps(
            {
                "version": summary["version"],
                "best_model": summary["best_model"],
                "primary_target": summary["primary_target"],
                "test_mae_score_points": exact["mae"],
                "test_mse": exact["mse"],
                "test_rmse_score_points": exact["rmse"],
                "test_r2": exact["r2"],
                "overall_level_accuracy_percent": 100
                * exact["health_level_accuracy"],
                "transition_level_accuracy_percent": 100
                * float(transition.get("health_level_accuracy") or 0.0),
                "critical_recall_percent": 100 * exact["critical_recall"],
                "deterioration_recall_percent": 100
                * metrics["deterioration"]["recall"],
                "forecast_bhsi_mae": forecast_indicators.get("forecast_bhsi_mae"),
                "forecast_bhsi_level_accuracy_percent": 100
                * float(
                    forecast_indicators.get("forecast_bhsi_level_accuracy")
                    or 0.0
                ),
                "forecast_rod_mae_points_per_hour": forecast_indicators.get(
                    "forecast_rod_mae"
                ),
                "forecast_trend_accuracy_percent": 100
                * float(
                    forecast_indicators.get("forecast_trend_accuracy") or 0.0
                ),
                "selected_score_weights": summary["weight_calibration"].get(
                    "selected_weights"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
