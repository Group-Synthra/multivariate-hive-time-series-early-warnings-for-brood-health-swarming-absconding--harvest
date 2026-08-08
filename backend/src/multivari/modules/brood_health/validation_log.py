from __future__ import annotations

import csv
import logging
import math
import os
import threading
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from .config import PATHS

logger = logging.getLogger(__name__)


class BroodForecastValidationLog:
    """Small project-local CSV log for prospective live forecast validation.

    The log intentionally stores only the values needed to reproduce the user's
    manual validation workflow: prediction time, current score, target time,
    predicted score, actual observation, error, and status.
    """

    FIELDNAMES: ClassVar[list[str]] = [
        "id",
        "device_id",
        "prediction_time",
        "current_score",
        "forecast_target_time",
        "predicted_score",
        "actual_time",
        "actual_score",
        "absolute_error",
        "status",
    ]

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or PATHS.validation_log_csv)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return

        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            writer.writeheader()

    @staticmethod
    def _clean(value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, float) and not math.isfinite(value):
            return ""

        return str(value)

    @staticmethod
    def _utc_iso(value: Any) -> str:
        timestamp = pd.Timestamp(value)

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        return timestamp.isoformat()

    def _read_rows(self) -> list[dict[str, str]]:
        self._ensure_file()

        with self.path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")

        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        field: self._clean(row.get(field))
                        for field in self.FIELDNAMES
                    }
                )

        os.replace(temp_path, self.path)

    def log_forecast(self, payload: dict[str, Any]) -> dict[str, Any]:
        required = (
            "device_id",
            "prediction_time",
            "current_score",
            "forecast_target_time",
            "predicted_score",
        )

        missing = [
            key for key in required if payload.get(key) in (None, "")
        ]

        if missing:
            raise ValueError(
                f"Validation log payload is missing required values: {missing}"
            )

        with self._lock:
            rows = self._read_rows()

            device_id = str(payload["device_id"])
            prediction_time = self._utc_iso(payload["prediction_time"])
            target_time = self._utc_iso(payload["forecast_target_time"])

            for row in rows:
                if (
                    row.get("device_id") == device_id
                    and row.get("prediction_time") == prediction_time
                    and row.get("forecast_target_time") == target_time
                ):
                    return dict(row)

            next_id = (
                max(
                    [
                        int(row.get("id", "0") or 0)
                        for row in rows
                        if str(row.get("id", "")).isdigit()
                    ],
                    default=0,
                )
                + 1
            )

            record = {
                "id": next_id,
                "device_id": device_id,
                "prediction_time": prediction_time,
                "current_score": float(payload["current_score"]),
                "forecast_target_time": target_time,
                "predicted_score": float(payload["predicted_score"]),
                "actual_time": "",
                "actual_score": "",
                "absolute_error": "",
                "status": "pending",
            }

            rows.append(record)
            self._write_rows(rows)

            return record

    def due(
        self,
        *,
        device_id: str | None = None,
        limit: int = 25,
        now: Any | None = None,
    ) -> list[dict[str, str]]:
        now_ts = (
            pd.Timestamp.now(tz="UTC")
            if now is None
            else pd.Timestamp(now)
        )

        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        else:
            now_ts = now_ts.tz_convert("UTC")

        with self._lock:
            rows = self._read_rows()

        due_rows: list[dict[str, str]] = []

        for row in rows:
            if row.get("status") != "pending":
                continue

            if device_id and row.get("device_id") != str(device_id):
                continue

            try:
                target = pd.Timestamp(row["forecast_target_time"])

                target = (
                    target.tz_localize("UTC")
                    if target.tzinfo is None
                    else target.tz_convert("UTC")
                )

            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping invalid forecast validation target "
                    "timestamp %r: %s",
                    row.get("forecast_target_time"),
                    exc,
                )
                continue

            if target <= now_ts:
                due_rows.append(row)

        due_rows.sort(
            key=lambda row: row.get("forecast_target_time", "")
        )

        return due_rows[: max(1, int(limit))]

    def complete(
        self,
        record_id: int,
        *,
        actual_time: Any,
        actual_score: float,
    ) -> None:
        with self._lock:
            rows = self._read_rows()

            for row in rows:
                if int(row.get("id", "0") or 0) != int(record_id):
                    continue

                predicted = float(row["predicted_score"])
                actual = float(actual_score)

                row["actual_time"] = self._utc_iso(actual_time)
                row["actual_score"] = actual
                row["absolute_error"] = abs(predicted - actual)
                row["status"] = "validated"

                break

            self._write_rows(rows)

    def summary(
        self,
        *,
        device_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        with self._lock:
            rows = self._read_rows()

        if device_id:
            rows = [
                row
                for row in rows
                if row.get("device_id") == str(device_id)
            ]

        validated = [
            row for row in rows if row.get("status") == "validated"
        ]

        pending = [
            row for row in rows if row.get("status") == "pending"
        ]

        errors: list[float] = []

        for row in validated:
            try:
                errors.append(float(row["absolute_error"]))
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping invalid absolute error value %r "
                    "for validation record %r",
                    row.get("absolute_error"),
                    row.get("id"),
                )

        recent = list(reversed(rows))[: max(1, int(limit))]

        return {
            "validated_forecasts": len(validated),
            "pending_forecasts": len(pending),
            "mae": (
                float(sum(errors) / len(errors))
                if errors
                else None
            ),
            "recent": recent,
            "file_name": self.path.name,
        }

    def csv_text(
        self,
        *,
        device_id: str | None = None,
    ) -> str:
        with self._lock:
            rows = self._read_rows()

        if device_id:
            rows = [
                row
                for row in rows
                if row.get("device_id") == str(device_id)
            ]

        stream = StringIO()

        writer = csv.DictWriter(
            stream,
            fieldnames=self.FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(rows)

        return stream.getvalue()