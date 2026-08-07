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


def main() -> None:
    load_dotenv(BACKEND_ROOT / ".env")
    settings = PostgresSensorSettings.from_env()
    repository = PostgresSensorRepository(settings)
    payload = repository.connectivity_payload()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
