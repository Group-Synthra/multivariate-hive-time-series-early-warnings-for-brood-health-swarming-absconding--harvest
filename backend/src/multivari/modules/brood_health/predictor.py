from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .analyzer import build_warning_payload, classify_health_level, compute_condition_history
from .config import PATHS
from .features import (
    FEATURE_SCHEMA_VERSION,
    SENSORS,
    aggregate_live_hourly,
    build_feature_frame,
    map_iot_frame,
    normalise_historical,
)
from .scoring import BroodHealthScoreConfig


class ModelNotReadyError(RuntimeError):
    pass


class BroodHealthPredictor:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = Path(model_path or PATHS.model_bundle)
        self._bundle: dict[str, Any] | None = None
        self._model_mtime: float | None = None

    def _load_bundle(self) -> dict[str, Any]:
        if not self.model_path.exists():
            raise ModelNotReadyError(
                "The brood-health model has not been trained. Run python scripts/train_brood_health.py first."
            )
        mtime = self.model_path.stat().st_mtime
        if self._bundle is None or self._model_mtime != mtime:
            bundle = joblib.load(self.model_path)
            required = {
                "model",
                "feature_columns",
                "horizon_hours",
                "training_sensor_reference",
                "score_config",
            }
            missing = sorted(required.difference(bundle))
            if missing:
                raise ModelNotReadyError(
                    "The saved brood-health model bundle is from an older incompatible version. "
                    f"Missing fields: {missing}. Retrain the model."
                )
            if bundle.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
                raise ModelNotReadyError(
                    "The saved model uses an incompatible feature schema. Retrain the brood-health model."
                )
            self._bundle = bundle
            self._model_mtime = mtime
        return self._bundle

    def model_info(self) -> dict[str, Any]:
        bundle = self._load_bundle()
        return {
            "model_name": bundle["model_name"],
            "trained_at_utc": bundle["trained_at_utc"],
            "horizon_hours": bundle["horizon_hours"],
            "feature_count": len(bundle["feature_columns"]),
            "target_column": bundle["target_column"],
            "target_kind": bundle["target_kind"],
            "target_range": bundle.get("target_range", [1.0, 100.0]),
        }

    @staticmethod
    def _domain_shift(latest: pd.Series, reference: dict[str, dict[str, float]]) -> list[str]:
        warnings: list[str] = []
        for sensor in SENSORS:
            value = latest.get(sensor)
            limits = reference.get(sensor, {})
            if value is None or pd.isna(value):
                warnings.append(f"{sensor} is missing in the latest hourly aggregate")
                continue
            if limits and (float(value) < float(limits["p01"]) or float(value) > float(limits["p99"])):
                warnings.append(
                    f"{sensor}={float(value):.2f} is outside the historical 1st–99th percentile range "
                    f"({float(limits['p01']):.2f} to {float(limits['p99']):.2f})"
                )
        return warnings

    @staticmethod
    def _feature_completeness(row: pd.Series) -> float:
        return float(row.notna().mean() * 100.0)

    def predict_hourly_history(self, hourly: pd.DataFrame) -> dict[str, Any]:
        bundle = self._load_bundle()
        data = normalise_historical(hourly)
        if data.empty:
            raise ValueError("No valid hourly IoT data are available for prediction")
        if data["hive_id"].nunique() != 1:
            raise ValueError("Live prediction accepts one device/hive history at a time")

        data = data.sort_values(["hive_id", "timestamp"]).reset_index(drop=True)
        data[list(SENSORS)] = data.groupby("hive_id", sort=False)[list(SENSORS)].ffill(limit=2)
        data = data.dropna(subset=list(SENSORS)).reset_index(drop=True)
        if data.empty:
            raise ValueError("No complete hourly sensor observation is available for prediction")

        feature_frame = build_feature_frame(data).reindex(columns=bundle["feature_columns"])
        model = bundle["model"]
        forecast_scores = np.clip(model.predict(feature_frame), 1.0, 100.0)

        score_config = BroodHealthScoreConfig.from_dict(bundle.get("score_config"))
        condition = compute_condition_history(data, score_config=score_config)
        result = condition[
            [
                "hive_id",
                "timestamp",
                *SENSORS,
                "condition_score",
                "condition_level",
                "bhsi",
                "stability_level",
                "rod_points_per_hour",
                "trend_label",
            ]
        ].copy()
        result["forecast_score"] = forecast_scores
        result["forecast_level"] = result["forecast_score"].map(classify_health_level)
        result["forecast_change_points"] = result["forecast_score"] - result["condition_score"]
        result["forecast_drop_points"] = (
            result["condition_score"] - result["forecast_score"]
        ).clip(lower=0.0)
        result["risk_index"] = (100.0 - result["forecast_score"]).clip(0.0, 99.0)

        latest = result.iloc[-1]
        latest_features = feature_frame.iloc[-1]
        domain_shift = self._domain_shift(latest, bundle["training_sensor_reference"])
        history_sufficient = len(data) >= 72
        warning = build_warning_payload(
            forecast_score=float(latest["forecast_score"]),
            current_condition_score=float(latest["condition_score"]),
            bhsi=float(latest["bhsi"]),
            rod_points_per_hour=float(latest["rod_points_per_hour"]),
            forecast_drop_points=float(latest["forecast_drop_points"]),
            domain_shift_warnings=domain_shift,
            history_sufficient=history_sufficient,
        )

        history = result.tail(168).copy()
        history["timestamp"] = history["timestamp"].map(lambda value: value.isoformat())
        history_records = history.to_dict(orient="records")
        for row in history_records:
            for key, value in list(row.items()):
                if isinstance(value, (np.integer,)):
                    row[key] = int(value)
                elif isinstance(value, (np.floating, float)):
                    row[key] = float(value) if np.isfinite(value) else None

        optional_context: dict[str, float | None] = {}
        for column in ("external_temp", "external_humidity", "battery_voltage", "raw_reading_count"):
            if column in data.columns:
                value = data.iloc[-1][column]
                optional_context[column] = None if pd.isna(value) else float(value)

        latest_timestamp = pd.Timestamp(latest["timestamp"])
        now = pd.Timestamp.now(tz="UTC")
        if latest_timestamp.tzinfo is None:
            latest_timestamp = latest_timestamp.tz_localize("UTC")
        freshness_minutes = max(0.0, float((now - latest_timestamp).total_seconds() / 60.0))

        return {
            "device_id": str(latest["hive_id"]),
            "latest_timestamp": latest_timestamp.isoformat(),
            "data_freshness_minutes": freshness_minutes,
            "hourly_rows": int(len(data)),
            "raw_readings_in_latest_hour": optional_context.get("raw_reading_count"),
            "feature_completeness_percentage": self._feature_completeness(latest_features),
            "minimum_recommended_history_hours": 72,
            "history_sufficiency": "good" if history_sufficient else "limited",
            "model": self.model_info(),
            "latest_sensors": {sensor: float(latest[sensor]) for sensor in SENSORS},
            "context": optional_context,
            "prediction": {
                "forecast_score": float(latest["forecast_score"]),
                "forecast_level": str(latest["forecast_level"]),
                "forecast_change_points": float(latest["forecast_change_points"]),
                "forecast_drop_points": float(latest["forecast_drop_points"]),
                "risk_index": float(latest["risk_index"]),
                "horizon_hours": int(bundle["horizon_hours"]),
                "target_kind": bundle["target_kind"],
                "interpretation": (
                    "Predicted minimum Brood Health Score expected within the forecast window."
                ),
            },
            "current_condition": {
                "score": float(latest["condition_score"]),
                "level": str(latest["condition_level"]),
                "bhsi": float(latest["bhsi"]),
                "stability_level": str(latest["stability_level"]),
                "rod_points_per_hour": float(latest["rod_points_per_hour"]),
                "trend_label": str(latest["trend_label"]),
            },
            "domain_shift_warnings": domain_shift,
            "warning": warning,
            "history": history_records,
            "score_definition": {
                "range": "1–100",
                "levels": ["Critical", "Poor", "Good", "Excellent"],
                "is_direct_biological_measurement": False,
            },
            "disclaimer": (
                "This is a sensor-based decision-support output. Confirm Poor or Critical results "
                "through physical brood inspection."
            ),
        }

    def predict_raw_iot(self, raw: pd.DataFrame) -> dict[str, Any]:
        mapped = map_iot_frame(raw)
        hourly = aggregate_live_hourly(mapped)
        return self.predict_hourly_history(hourly)
