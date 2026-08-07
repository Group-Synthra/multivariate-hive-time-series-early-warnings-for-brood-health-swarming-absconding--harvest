from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = BACKEND_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from multivari.iot.postgres_repository import (
    PostgresSensorRepository,
    PostgresSensorSettings,
)
from multivari.modules.harvesting.live_hui_inference import (
    LiveHuiInferenceEngine,
)


def main() -> None:
    load_dotenv(BACKEND_ROOT / ".env")
    sensor_settings = PostgresSensorSettings.from_env()
    repository = PostgresSensorRepository(sensor_settings)
    raw = repository.fetch_recent()
    engine = LiveHuiInferenceEngine(
        backend_root=BACKEND_ROOT,
        sensor_settings=sensor_settings,
    )
    payload = engine.build_payload(raw)
    summary = {
        "status": payload["status"],
        "generated_at": payload["generated_at"],
        "available_hives": payload["available_hives"],
        "latest_by_hive": payload["latest_by_hive"],
        "hive_diagnostics": payload["hive_diagnostics"],
        "research_status": payload["research_status"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
