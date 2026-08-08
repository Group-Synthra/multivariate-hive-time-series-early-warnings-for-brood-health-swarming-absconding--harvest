from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from flask import Blueprint, jsonify, request


forecast_validation_bp = Blueprint("forecast_validation", __name__)

DATA_FILE = Path(
    os.getenv(
        "SWARMING_FORECAST_VALIDATION_FILE",
        Path(__file__).resolve().parent / "data" / "swarming_forecast_validation.json",
    )
)
MATCH_TOLERANCE = timedelta(minutes=10)
VALID_FORECAST_DAYS = {1, 2, 3}
_FILE_LOCK = threading.Lock()


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_rows() -> list[dict[str, Any]]:
    if not DATA_FILE.exists():
        return []
    with DATA_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else []


def _save_rows(rows: list[dict[str, Any]]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=DATA_FILE.parent, delete=False, suffix=".tmp"
    ) as temp_file:
        json.dump(rows, temp_file, indent=2, ensure_ascii=False)
        temp_path = Path(temp_file.name)
    temp_path.replace(DATA_FILE)


def _risk_level(value: float) -> str:
    if value >= 60:
        return "HIGH"
    if value >= 30:
        return "MEDIUM"
    return "LOW"


def _round_risk(value: Any) -> float:
    risk = float(value)
    if not 0 <= risk <= 100:
        raise ValueError("Risk percentage must be between 0 and 100")
    return round(risk, 4)


def _normalise_forecasts(forecasts: Any) -> list[dict[str, Any]]:
    if not isinstance(forecasts, list):
        raise ValueError("forecasts must be a list")

    normalised: list[dict[str, Any]] = []
    seen_days: set[int] = set()
    for item in forecasts:
        if not isinstance(item, dict):
            raise ValueError("Each forecast must be an object")
        day = int(item.get("day"))
        if day not in VALID_FORECAST_DAYS:
            raise ValueError("Forecast day must be 1, 2, or 3")
        if day in seen_days:
            raise ValueError(f"Duplicate forecast day: {day}")
        seen_days.add(day)
        normalised.append({"day": day, "risk": _round_risk(item.get("risk"))})

    if seen_days != VALID_FORECAST_DAYS:
        raise ValueError("Forecasts must contain Day 1, Day 2, and Day 3")
    return sorted(normalised, key=lambda item: item["day"])


def _migrate_old_day1_rows(rows: list[dict[str, Any]]) -> None:
    """Keep JSON records created by the earlier Day-1-only version usable."""
    for row in rows:
        if "forecast_risk" not in row and "day1_forecast_risk" in row:
            row["forecast_risk"] = row.pop("day1_forecast_risk")
        if "forecast_level" not in row and "day1_forecast_level" in row:
            row["forecast_level"] = row.pop("day1_forecast_level")
        row.setdefault("forecast_day", 1)


def _validate_pending_rows(
    rows: list[dict[str, Any]],
    *,
    device_id: str,
    observation_time: datetime,
    current_risk: float,
) -> None:
    """Validate the closest pending row for each Day-1/Day-2/Day-3 horizon."""
    for forecast_day in sorted(VALID_FORECAST_DAYS):
        candidates: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            if (
                row.get("device_id") != device_id
                or row.get("status") != "PENDING"
                or int(row.get("forecast_day", 1)) != forecast_day
            ):
                continue

            difference_seconds = abs(
                (observation_time - _parse_timestamp(row["target_at"])).total_seconds()
            )
            if difference_seconds <= MATCH_TOLERANCE.total_seconds():
                candidates.append((difference_seconds, row))

        if not candidates:
            continue

        difference_seconds, matched = min(candidates, key=lambda item: item[0])
        forecast_value = float(matched["forecast_risk"])
        signed_error = round(current_risk - forecast_value, 4)
        matched.update(
            {
                "actual_observed_at": _iso(observation_time),
                "actual_current_risk": current_risk,
                "actual_current_level": _risk_level(current_risk),
                "time_difference_minutes": round(difference_seconds / 60, 2),
                "signed_error": signed_error,
                "absolute_error": round(abs(signed_error), 4),
                "level_match": matched["forecast_level"]
                == _risk_level(current_risk),
                "status": "VALIDATED",
                "actual_swarm_event": None,
            }
        )


def record_and_match(
    *,
    device_id: str,
    observed_at: str,
    current_risk: Any,
    forecasts: Any,
) -> list[dict[str, Any]]:
    observation_time = _parse_timestamp(observed_at)
    current = _round_risk(current_risk)
    clean_forecasts = _normalise_forecasts(forecasts)
    made_at = _iso(observation_time)

    with _FILE_LOCK:
        rows = _load_rows()
        _migrate_old_day1_rows(rows)

        # First use today's current risk to validate forecasts made 1/2/3 days ago.
        _validate_pending_rows(
            rows,
            device_id=device_id,
            observation_time=observation_time,
            current_risk=current,
        )

        # Then save today's Day-1, Day-2 and Day-3 forecasts as separate rows.
        for forecast in clean_forecasts:
            forecast_day = forecast["day"]
            already_saved = any(
                row.get("device_id") == device_id
                and row.get("forecast_made_at") == made_at
                and int(row.get("forecast_day", 1)) == forecast_day
                for row in rows
            )
            if already_saved:
                continue

            risk = forecast["risk"]
            target_time = observation_time + timedelta(days=forecast_day)
            rows.append(
                {
                    "device_id": device_id,
                    "forecast_day": forecast_day,
                    "forecast_made_at": made_at,
                    "target_at": _iso(target_time),
                    "forecast_risk": risk,
                    "forecast_level": _risk_level(risk),
                    "actual_observed_at": None,
                    "actual_current_risk": None,
                    "actual_current_level": None,
                    "time_difference_minutes": None,
                    "signed_error": None,
                    "absolute_error": None,
                    "level_match": None,
                    "actual_swarm_event": None,
                    "status": "PENDING",
                }
            )

        rows.sort(
            key=lambda row: (
                row.get("forecast_made_at", ""),
                int(row.get("forecast_day", 1)),
            )
        )
        _save_rows(rows)
        return [row for row in rows if row.get("device_id") == device_id]


@forecast_validation_bp.post("/api/swarming/forecast-validation")
def save_forecast_validation():
    payload = request.get_json(silent=True) or {}
    required = ("device_id", "observed_at", "current_risk", "forecasts")
    missing = [key for key in required if payload.get(key) is None]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        rows = record_and_match(
            device_id=str(payload["device_id"]),
            observed_at=str(payload["observed_at"]),
            current_risk=payload["current_risk"],
            forecasts=payload["forecasts"],
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        return jsonify({"error": str(error)}), 400

    return jsonify({"history": rows[-1500:], "count": len(rows)})


@forecast_validation_bp.get("/api/swarming/forecast-validation")
def get_forecast_validation():
    device_id = request.args.get("device_id", "").strip()
    with _FILE_LOCK:
        rows = _load_rows()
        _migrate_old_day1_rows(rows)
    if device_id:
        rows = [row for row in rows if row.get("device_id") == device_id]
    return jsonify({"history": rows[-1500:], "count": len(rows)})