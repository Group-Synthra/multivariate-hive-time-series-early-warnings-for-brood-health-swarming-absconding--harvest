from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from multivari.iot.database import get_engine


class SwarmingLiveService:
    """Provide live swarming predictions using IoT readings."""

    def __init__(self) -> None:
        self._predictor_module = None
        self._risk_classifier = None

    def _load(self):
        if self._predictor_module is None:
            from .live_prediction import live_predictor
            from .live_prediction.risk_classifier import create_risk_classifier

            self._predictor_module = live_predictor
            self._risk_classifier = create_risk_classifier()
        return self._predictor_module, self._risk_classifier

    def health(self) -> dict[str, bool]:
        from .live_prediction.config import (
            LABEL_ENCODER_PATH,
            LSTM_MODEL_PATH,
            SCALER_PATH,
        )

        status = {
            "lstm_model_ready": os.path.exists(LSTM_MODEL_PATH),
            "scaler_ready": os.path.exists(SCALER_PATH),
            "label_encoder_ready": os.path.exists(LABEL_ENCODER_PATH),
        }
        status["all_ready"] = all(status.values())
        return status

    def sample_payload(self, count: int = 24) -> dict[str, Any]:
        return {"hive_id": "Hive_01", "readings": self._generate_sample_readings(count)}

    def predict(self, hive_id: str, readings: list[dict[str, float]]) -> dict[str, Any]:
        predictor, classifier = self._load()
        result = predictor.predict(hive_id, readings)
        classification = classifier.classify_from_probability(result.get("probability", 0))
        result["risk_percentage"] = classification["risk_percentage"]
        result["risk_level"] = classification["risk_level"]
        result["warning"] = classification["message"]
        result["event_window"] = classification["event_window"]
        result["softmax_probabilities"] = classification.get("softmax_probabilities", {})
        return result

    def predict_from_iot(self, device_id: str, limit: int = 432) -> dict[str, Any]:
        """Return a prediction plus timestamped sensor history for the live chart."""
        table = self._safe_identifier("IOT_SENSOR_TABLE", "beehive_readings")
        hive_col = self._safe_identifier("IOT_HIVE_COLUMN", "device_id")
        time_col = self._safe_identifier("IOT_TIMESTAMP_COLUMN", "recorded_at")
        columns = {
            "internal_temp": self._safe_identifier("IOT_TEMPERATURE_COLUMN", "internal_temp"),
            "internal_humidity": self._safe_identifier("IOT_HUMIDITY_COLUMN", "internal_humidity"),
            "internal_co2": self._safe_identifier("IOT_CO2_COLUMN", "internal_co2"),
            "total_weight": self._safe_identifier("IOT_WEIGHT_COLUMN", "total_weight"),
            "external_temp": self._safe_identifier("IOT_EXTERNAL_TEMPERATURE_COLUMN", "external_temp"),
            "external_humidity": self._safe_identifier("IOT_EXTERNAL_HUMIDITY_COLUMN", "external_humidity"),
            "battery_voltage": self._safe_identifier("IOT_BATTERY_VOLTAGE_COLUMN", "battery_voltage"),
        }
        selections = [
            f"{time_col} AS recorded_at",
            *[f"{source} AS {alias}" for alias, source in columns.items()],
        ]
        query = text(
            f"SELECT {', '.join(selections)} FROM {table} WHERE {hive_col} = :device "
            f"ORDER BY {time_col} DESC LIMIT :limit"
        )
        frame = pd.read_sql(
            query,
            get_engine(),
            params={"device": device_id, "limit": max(24, limit)},
        ).sort_values("recorded_at")

        if len(frame) < 24:
            raise ValueError(f"Not enough data: only {len(frame)} readings found (need at least 24).")

        latest = frame.iloc[-1]
        latest_sensor = {
            "internal_temperature_c": self._number(latest["internal_temp"]),
            "internal_humidity_pct": self._number(latest["internal_humidity"]),
            "co2_ppm": self._number(latest["internal_co2"]),
            "hive_weight_kg": self._number(latest["total_weight"]),
            "external_temperature_c": self._number(latest["external_temp"]),
            "external_humidity_pct": self._number(latest["external_humidity"]),
            "battery_voltage": self._number(latest["battery_voltage"]),
            "recorded_at": str(latest["recorded_at"]),
        }

        all_readings = [
            {
                "recorded_at": str(row["recorded_at"]),
                "internal_temperature_c": self._number(row["internal_temp"], 35.0),
                "internal_humidity_pct": self._number(row["internal_humidity"], 65.0),
                "co2_ppm": self._number(row["internal_co2"], 1200.0),
                "hive_weight_kg": self._number(row["total_weight"], 32.5),
                "external_temperature_c": self._number(row["external_temp"], 28.0),
                "external_humidity_pct": self._number(row["external_humidity"], 55.0),
                "rainfall_mm_hour": 0.0,
                "wind_speed_mps": 0.0,
            }
            for _, row in frame.iterrows()
        ]

        prediction_readings = all_readings[-24:]
        prediction = self.predict(device_id, prediction_readings)

        current_risk = float(prediction.get("risk_percentage", 0.0))
        temp_values = np.array(
            [item["internal_temperature_c"] for item in prediction_readings]
        )
        temp_trend = float(np.polyfit(np.arange(len(temp_values)), temp_values, 1)[0])
        classifier = self._load()[1]
        forecast = []
        for day in range(1, 4):
            daily_cycle = 7.5 + 7.5 * np.sin(
                ((datetime.now().hour + day * 24) % 24 - 6) * np.pi / 12
            )
            projected = float(
                np.clip(
                    0.7 * current_risk
                    + 0.2 * (current_risk + temp_trend * day * 5)
                    + 0.1 * daily_cycle,
                    0,
                    100,
                )
            )
            classified = classifier.classify_from_probability(projected / 100)
            forecast.append(
                {
                    "day": day,
                    "risk": round(projected, 1),
                    "level": classified["risk_level"],
                    "color": classifier.get_risk_color(classified["risk_level"]),
                }
            )

        return {
            "device_id": device_id,
            "latest_sensor": latest_sensor,
            "prediction": prediction,
            "forecast": forecast,
            "readings_used": len(prediction_readings),
            "readings_fetched": len(all_readings),
            "readings": all_readings,
            "event_window": prediction.get("event_window", {}),
            "key_factors": prediction.get("key_factors", []),
            "recommendations": prediction.get("recommendations", []),
            "softmax_probabilities": prediction.get("softmax_probabilities", {}),
            "formula_used": prediction.get("formula_used", ""),
        }

    @staticmethod
    def _safe_identifier(env_name: str, default: str) -> str:
        value = os.getenv(env_name, default)
        if not value.replace("_", "").isalnum():
            raise ValueError(f"Unsafe SQL identifier configured for {env_name}")
        return value

    @staticmethod
    def _number(value, default: float = 0.0) -> float:
        return round(float(value), 2) if pd.notna(value) else default

    @staticmethod
    def _generate_sample_readings(n: int = 24) -> list[dict[str, float]]:
        """Synthetic readings that mimic a pre-swarm state."""
        random.seed(42)
        readings = []
        for i in range(n):
            temp_trend = 0.3 * i
            co2_trend = 15 * i
            weight_drop = -0.05 * i
            readings.append(
                {
                    "internal_temperature_c": round(34.5 + temp_trend + random.uniform(-0.2, 0.2), 2),
                    "internal_humidity_pct": round(65.0 + random.uniform(-1.0, 1.0), 2),
                    "co2_ppm": round(1200 + co2_trend + random.uniform(-30, 30), 2),
                    "hive_weight_kg": round(35.0 + weight_drop + random.uniform(-0.1, 0.1), 2),
                    "external_temperature_c": round(28.0 + random.uniform(-0.5, 0.5), 2),
                    "external_humidity_pct": round(55.0 + random.uniform(-1.0, 1.0), 2),
                    "rainfall_mm_hour": round(max(0, random.uniform(0, 0.2)), 2),
                    "wind_speed_mps": round(2.0 + random.uniform(0, 0.5), 2),
                }
            )
        return readings