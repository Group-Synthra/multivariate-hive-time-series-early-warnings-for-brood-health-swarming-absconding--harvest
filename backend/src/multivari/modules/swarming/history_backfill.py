from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from .risk_history_json import get_risk_history, save_risk_prediction

logger = logging.getLogger(__name__)

MINIMUM_MODEL_READINGS = 144
BACKFILL_HOURS = 24


def _number(value: Any, default: float) -> float:
    """Convert a database value to float, using a safe default."""
    if pd.notna(value):
        return float(value)
    return default


def _frame_to_readings(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert aliased IoT database rows to predictor input records."""
    return [
        {
            "recorded_at": row["recorded_at"].isoformat(),
            "internal_temperature_c": _number(row["internal_temp"], 35.0),
            "internal_humidity_pct": _number(row["internal_humidity"], 65.0),
            "co2_ppm": _number(row["internal_co2"], 1200.0),
            "hive_weight_kg": _number(row["total_weight"], 32.5),
            "external_temperature_c": _number(row["external_temp"], 28.0),
            "external_humidity_pct": _number(row["external_humidity"], 55.0),
            "rainfall_mm_hour": 0.0,
            "wind_speed_mps": 0.0,
        }
        for _, row in frame.iterrows()
    ]


def _timestamp_key(value: Any) -> int | None:
    """Return one UTC nanosecond key for reliable timestamp comparison."""
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return int(timestamp.value)


def backfill_missing_risks(
    *,
    device_id: str,
    frame: pd.DataFrame,
    predictor: Any,
    risk_classifier: Any,
) -> dict[str, int]:
    """Calculate missing 10-minute risks from stored historical IoT rows.

    Only real timestamps present in the IoT database are considered. Existing
    JSON records are skipped, so normal refreshes do not repeat model work.
    """
    required_columns = {
        "recorded_at",
        "internal_temp",
        "internal_humidity",
        "internal_co2",
        "total_weight",
        "external_temp",
        "external_humidity",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Backfill frame is missing columns: {names}")

    prepared = frame.copy()
    prepared["recorded_at"] = pd.to_datetime(
        prepared["recorded_at"],
        utc=True,
        errors="coerce",
    )
    prepared = (
        prepared.dropna(subset=["recorded_at"])
        .sort_values("recorded_at")
        .drop_duplicates(subset=["recorded_at"], keep="last")
        .reset_index(drop=True)
    )

    if prepared.empty:
        return {"candidates": 0, "created": 0, "skipped": 0, "failed": 0}

    cutoff = datetime.now(UTC) - timedelta(hours=BACKFILL_HOURS)
    candidates = prepared.loc[prepared["recorded_at"] >= cutoff, "recorded_at"]

    stored_history = get_risk_history(
        device_id=device_id,
        minutes=BACKFILL_HOURS * 60,
        limit=5000,
    )
    stored_keys = {
        key
        for item in stored_history
        if (key := _timestamp_key(item.get("predicted_at"))) is not None
    }

    created = 0
    skipped = 0
    failed = 0

    for candidate_timestamp in candidates:
        candidate_key = _timestamp_key(candidate_timestamp)
        if candidate_key is None:
            failed += 1
            continue

        if candidate_key in stored_keys:
            skipped += 1
            continue

        historical_frame = prepared.loc[prepared["recorded_at"] <= candidate_timestamp]
        if len(historical_frame) < MINIMUM_MODEL_READINGS:
            skipped += 1
            continue

        readings = _frame_to_readings(historical_frame)

        try:
            prediction = predictor.predict(device_id, readings)
            combination = risk_classifier.combine_lstm_and_pelt(
                lstm_probability=prediction.get("probability", 0),
                pelt_snapshot=prediction.get("pelt_snapshot", {}),
            )
            classification = risk_classifier.classify_from_probability(
                combination["combined_probability"]
            )

            prediction.update(combination)
            prediction["combined_probability"] = combination["combined_probability"]
            prediction["risk_percentage"] = combination["risk_percentage"]
            prediction["risk_level"] = classification["risk_level"]
            prediction["predicted_class"] = (
                "Swarming"
                if combination["combined_probability"] >= prediction.get("threshold_used", 0.70)
                else "No Swarming"
            )
            prediction["data_timestamp"] = candidate_timestamp.isoformat()

            save_risk_prediction(
                device_id=device_id,
                prediction=prediction,
            )
            stored_keys.add(candidate_key)
            created += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.warning(
                "Could not backfill risk for %s at %s: %s",
                device_id,
                candidate_timestamp.isoformat(),
                exc,
            )

    return {
        "candidates": len(candidates),
        "created": created,
        "skipped": skipped,
        "failed": failed,
    }
