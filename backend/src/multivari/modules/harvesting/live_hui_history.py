from __future__ import annotations

import csv
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_WRITE_LOCK = threading.Lock()

FIELD_NAMES = [
    "saved_at_utc",
    "timestamp",
    "hive_id",
    "current_hui",
    "current_class",
    "predicted_hui_24h",
    "predicted_class_24h",
    "predicted_hui_48h",
    "predicted_class_48h",
    "predicted_hui_72h",
    "predicted_class_72h",
    "hrsi",
    "hrsi_interpretation",
    "rate_of_change",
    "rate_of_change_points_per_hour",
    "confidence_score",
    "prediction_confidence",
    "recommended_window",
    "used_interpolation",
]

def append_live_hui_prediction(
    csv_path: Path,
    prediction: dict[str, Any],
) -> None:
    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = {
    field: prediction.get(field)
    for field in FIELD_NAMES
    }

    row["saved_at_utc"] = datetime.now(UTC).isoformat()

    file_is_new = (
        not csv_path.exists()
        or csv_path.stat().st_size == 0
    )

    with _WRITE_LOCK, csv_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELD_NAMES,
        )

        if file_is_new:
            writer.writeheader()

        writer.writerow(row)


def read_live_hui_history(
    csv_path: Path,
    *,
    hive_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []

    with _WRITE_LOCK, csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    if hive_id:
        rows = [
            row
            for row in rows
            if str(row.get("hive_id")) == str(hive_id)
        ]

    return rows[-limit:][::-1]
