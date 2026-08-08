from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
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
    HIVE_COLUMN,
    TIMESTAMP_COLUMN,
    LiveHuiInferenceEngine,
)


def _gap_records(group: pd.DataFrame) -> list[dict[str, object]]:
    ordered = group.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    elapsed = ordered[TIMESTAMP_COLUMN].diff().dt.total_seconds().div(3600.0)
    records: list[dict[str, object]] = []
    for index in elapsed.loc[elapsed.gt(1.0)].index:
        previous = ordered.loc[index - 1, TIMESTAMP_COLUMN]
        current = ordered.loc[index, TIMESTAMP_COLUMN]
        gap_hours = float(elapsed.loc[index])
        records.append(
            {
                "previous_hour": previous.isoformat(),
                "next_hour": current.isoformat(),
                "elapsed_hours": gap_hours,
                "missing_hour_count": max(0, round(gap_hours) - 1),
            }
        )
    return records


def main() -> None:
    load_dotenv(BACKEND_ROOT / ".env")
    settings = PostgresSensorSettings.from_env()
    repository = PostgresSensorRepository(settings)
    raw = repository.fetch_recent()
    engine = LiveHuiInferenceEngine(
        backend_root=BACKEND_ROOT,
        sensor_settings=settings,
    )
    hourly, _, diagnostics = engine.prepare_hourly_history(raw)

    payload: dict[str, object] = {
        "status": "live_hui_hourly_gap_report",
        "history_hours": settings.history_hours,
        "history_reference": settings.history_reference,
        "hives": [],
    }
    diagnostics_by_hive = {item["hive_id"]: item for item in diagnostics}

    for hive_id, group in hourly.groupby(HIVE_COLUMN, sort=True):
        hive_key = str(hive_id)
        gaps = _gap_records(group)
        diagnostic = diagnostics_by_hive[hive_key]
        payload["hives"].append(
            {
                "hive_id": hive_key,
                "first_hour": group[TIMESTAMP_COLUMN].min().isoformat(),
                "latest_hour": group[TIMESTAMP_COLUMN].max().isoformat(),
                "hourly_rows": len(group),
                "latest_contiguous_hourly_rows": diagnostic["latest_contiguous_hourly_rows"],
                "hours_needed_for_current_hui": max(
                    0,
                    168 - int(diagnostic["latest_contiguous_hourly_rows"]),
                ),
                "hours_needed_for_future_hui": max(
                    0,
                    192 - int(diagnostic["latest_contiguous_hourly_rows"]),
                ),
                "gap_count": len(gaps),
                "gaps": gaps[-20:],
            }
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
