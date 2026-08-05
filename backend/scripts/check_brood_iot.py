from __future__ import annotations

import argparse
import json

from multivari.modules.brood_health.service import BroodHealthService


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Supabase/PostgreSQL brood-health IoT integration.")
    parser.add_argument("--device-id", help="Optionally run a complete prediction for one device.")
    parser.add_argument("--lookback-hours", type=int, default=None)
    args = parser.parse_args()

    service = BroodHealthService()
    status = service.iot_health()
    print(json.dumps(status, indent=2))

    if not status.get("database", {}).get("connected"):
        raise SystemExit(
            "Database check failed. Review backend/.env and the error shown above."
        )
    if args.device_id:
        result = service.predict_device(args.device_id, lookback_hours=args.lookback_hours)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(service.list_devices(), indent=2))


if __name__ == "__main__":
    main()
