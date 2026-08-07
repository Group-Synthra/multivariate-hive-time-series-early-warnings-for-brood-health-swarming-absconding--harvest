from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

HISTORY_FILE = Path(__file__).resolve().parent / "data" / "swarming_risk_history.json"

_file_lock = Lock()


def _read_history() -> list[dict[str, Any]]:
    """Read all stored swarming-risk records."""
    if not HISTORY_FILE.exists():
        return []

    try:
        with HISTORY_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_history(
    history: list[dict[str, Any]],
) -> None:
    """Safely write history using a temporary file."""
    HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = HISTORY_FILE.with_suffix(".tmp")

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(
        temporary_file,
        HISTORY_FILE,
    )


def _parse_timestamp(
    value: str,
) -> datetime | None:
    """Parse an ISO timestamp and convert it to UTC."""
    try:
        parsed = datetime.fromisoformat(value)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)

        return parsed.astimezone(UTC)
    except (TypeError, ValueError):
        return None


def save_risk_prediction(
    device_id: str,
    prediction: dict[str, Any],
) -> None:
    """Save or update a risk for one IoT data timestamp."""
    predicted_at = (
        prediction.get("data_timestamp")
        or prediction.get("timestamp")
        or prediction.get("predicted_at")
    )

    # Do not store a prediction without an IoT data timestamp.
    if not predicted_at:
        return

    parsed_prediction_time = _parse_timestamp(str(predicted_at))

    if parsed_prediction_time is None:
        return

    normalized_timestamp = parsed_prediction_time.isoformat()

    record = {
        "device_id": device_id,
        "predicted_at": normalized_timestamp,
        "lstm_risk_percentage": float(
            prediction.get(
                "lstm_risk_percentage",
                prediction.get(
                    "risk_percentage",
                    0,
                ),
            )
        ),
        "pelt_risk_percentage": float(
            prediction.get(
                "pelt_risk_percentage",
                0,
            )
        ),
        "combined_risk_percentage": float(
            prediction.get(
                "combined_risk_percentage",
                prediction.get(
                    "risk_percentage",
                    0,
                ),
            )
        ),
        "risk_level": prediction.get(
            "risk_level",
            "LOW",
        ),
        "predicted_class": prediction.get(
            "predicted_class",
            "No Swarming",
        ),
    }

    with _file_lock:
        history = _read_history()

        # Retain only records from the latest 24 hours.
        cutoff = datetime.now(UTC) - timedelta(hours=24)

        history = [
            item
            for item in history
            if (
                (
                    timestamp := _parse_timestamp(
                        str(
                            item.get(
                                "predicted_at",
                                "",
                            )
                        )
                    )
                )
                is not None
                and timestamp >= cutoff
            )
        ]

        # Check whether this exact IoT timestamp already exists.
        existing_index = next(
            (
                index
                for index, item in enumerate(history)
                if (
                    item.get("device_id") == device_id
                    and item.get("predicted_at") == normalized_timestamp
                )
            ),
            None,
        )

        if existing_index is None:
            # New IoT reading: add a new timeline point.
            history.append(record)
        else:
            # Same IoT reading: update the existing point.
            # Refresh Now will not create a duplicate.
            history[existing_index] = record

        history.sort(
            key=lambda item: item.get(
                "predicted_at",
                "",
            )
        )

        _write_history(history)


def get_risk_history(
    device_id: str,
    minutes: int = 1440,
    limit: int = 5000,
) -> list[dict[str, Any]]:
    """Return risk history for one device."""
    minutes = max(
        1,
        min(minutes, 1440),
    )
    limit = max(
        1,
        min(limit, 5000),
    )

    cutoff = datetime.now(UTC) - timedelta(minutes=minutes)

    with _file_lock:
        history = _read_history()

    filtered_history = []

    for item in history:
        timestamp = _parse_timestamp(
            str(
                item.get(
                    "predicted_at",
                    "",
                )
            )
        )

        if timestamp is not None and timestamp >= cutoff and item.get("device_id") == device_id:
            filtered_history.append(item)

    filtered_history.sort(
        key=lambda item: item.get(
            "predicted_at",
            "",
        )
    )

    return filtered_history[-limit:]
