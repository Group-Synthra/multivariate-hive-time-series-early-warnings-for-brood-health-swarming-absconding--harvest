from __future__ import annotations

import argparse
import json

from multivari.modules.brood_health.service import BroodHealthService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check PostgreSQL connectivity and the Brood Health v6 exact +6-hour predictor."
    )
    parser.add_argument("--device-id", help="Run a complete prediction for one live hive.")
    parser.add_argument("--lookback-hours", type=int, default=None)
    args = parser.parse_args()

    service = BroodHealthService()
    status = service.iot_health()
    print(json.dumps(status, indent=2))

    if not status.get("database", {}).get("connected"):
        raise SystemExit(
            "Database check failed. Review backend/.env, install psycopg and inspect the error above."
        )
    if not status.get("model", {}).get("ready"):
        raise SystemExit(
            "The Brood Health v6 model is not ready. Run scripts/train_brood_health.py first."
        )

    if args.device_id:
        result = service.predict_device(
            args.device_id,
            lookback_hours=args.lookback_hours,
        )
        concise = {
            "device_id": result["device_id"],
            "latest_raw_timestamp": result.get("live_latest_timestamp"),
            "rolling_forecast_anchor": result["prediction"].get("forecast_anchor_timestamp"),
            "forecast_timestamp": result["forecast_timestamp"],
            "observed_reading_interval_minutes": result.get("reading_interval_minutes"),
            "current_condition": result["current_condition"],
            "exact_plus_6h_prediction": {
                "score": result["prediction"]["exact_score"],
                "level": result["prediction"]["exact_level"],
                "interval_80": result["prediction"]["prediction_interval_80"],
                "interval_90": result["prediction"]["prediction_interval_90"],
            },
            "secondary_safety_minimum": {
                "score": result["prediction"]["safety_minimum_score"],
                "level": result["prediction"]["safety_minimum_level"],
            },
            "forecast_indicators": result.get("forecast_indicators", {}),
            "deployment": result.get("deployment", {}),
            "warning": result["warning"],
            "domain_shift_warnings": result.get("domain_shift_warnings", []),
            "weight_conversion": result.get("database_weight_conversion", {}),
        }
        print(json.dumps(concise, indent=2))
    else:
        print(json.dumps(service.list_devices(), indent=2))


if __name__ == "__main__":
    main()
