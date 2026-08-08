from __future__ import annotations

import csv

from multivari.modules.brood_health.validation_log import (
    BroodForecastValidationLog,
)


def test_local_validation_log_round_trip(tmp_path):
    path = tmp_path / "live_forecast_validation.csv"
    log = BroodForecastValidationLog(path)

    first = log.log_forecast(
        {
            "device_id": "hive_01",
            "prediction_time": "2026-08-08T02:23:00+00:00",
            "current_score": 49.3,
            "forecast_target_time": "2026-08-08T08:23:00+00:00",
            "predicted_score": 46.4,
        }
    )
    duplicate = log.log_forecast(
        {
            "device_id": "hive_01",
            "prediction_time": "2026-08-08T02:23:00+00:00",
            "current_score": 49.3,
            "forecast_target_time": "2026-08-08T08:23:00+00:00",
            "predicted_score": 46.4,
        }
    )

    assert first["id"] == 1
    assert str(duplicate["id"]) == "1"
    assert len(log.due(now="2026-08-08T08:24:00+00:00")) == 1

    log.complete(
        1,
        actual_time="2026-08-08T08:27:00+00:00",
        actual_score=75.55,
    )

    summary = log.summary(device_id="hive_01")
    assert summary["validated_forecasts"] == 1
    assert summary["pending_forecasts"] == 0
    assert round(summary["mae"], 2) == 29.15

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0].keys()) == BroodForecastValidationLog.FIELDNAMES
    assert rows[0]["status"] == "validated"
    assert float(rows[0]["current_score"]) == 49.3
    assert float(rows[0]["predicted_score"]) == 46.4
    assert float(rows[0]["actual_score"]) == 75.55
