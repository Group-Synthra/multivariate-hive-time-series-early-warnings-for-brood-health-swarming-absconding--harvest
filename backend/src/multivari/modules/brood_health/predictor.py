from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .analyzer import (
    build_warning_payload,
    classify_stability,
    classify_trend,
    compute_condition_history,
)
from .config import PATHS
from .features import (
    FEATURE_SCHEMA_VERSION,
    SENSORS,
    aggregate_live_hourly,
    build_feature_frame,
    map_iot_frame,
    normalise_historical,
)
from .forecast_indicators import (
    forecast_bhsi,
    forecast_rod,
    interpolate_forecast_trajectory,
)
from .scoring import BroodHealthScoreConfig, classify_health_level


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
                "The Brood Health v6 model has not been trained. Run "
                "python scripts/train_brood_health.py --horizon-hours 6 first."
            )
        mtime = self.model_path.stat().st_mtime
        if self._bundle is None or self._model_mtime != mtime:
            bundle = joblib.load(self.model_path)
            required = {
                "model",
                "model_name",
                "feature_columns",
                "feature_schema_version",
                "horizon_hours",
                "target_columns",
                "score_config",
                "training_sensor_reference",
                "prediction_interval_absolute_error",
                "forecast_stability_reference",
                "forecast_indicator_definition",
            }
            missing = sorted(required.difference(bundle))
            if missing:
                raise ModelNotReadyError(
                    "The saved model is from an older incompatible version. "
                    f"Missing fields: {missing}. Run the cleanup script and retrain v6."
                )
            if bundle["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
                raise ModelNotReadyError(
                    "The saved model uses an incompatible feature schema. "
                    "Delete generated Brood Health artifacts and retrain v6."
                )
            self._bundle = bundle
            self._model_mtime = mtime
        return self._bundle

    def model_info(self) -> dict[str, Any]:
        bundle = self._load_bundle()
        return {
            "version": "6.0",
            "model_name": bundle["model_name"],
            "trained_at_utc": bundle["trained_at_utc"],
            "horizon_hours": int(bundle["horizon_hours"]),
            "feature_count": len(bundle["feature_columns"]),
            "primary_target": bundle["primary_target"],
            "target_kind": bundle["target_kind"],
            "target_range": bundle.get("target_range", [1.0, 100.0]),
            "forecast_horizons": list(
                range(1, int(bundle["horizon_hours"]) + 1)
            ),
            "native_model_resolution_minutes": 60,
            "live_anchor_strategy": (
                "rolling one-hour medians aligned to the latest raw IoT reading"
            ),
            "weight_transfer_strategy": bundle.get(
                "weight_transfer_strategy",
                "relative_change_and_stability_only",
            ),
        }

    @staticmethod
    def _feature_completeness(row: pd.Series) -> float:
        return float(row.notna().mean() * 100.0)

    @staticmethod
    def _environment_domain_shift(
        latest: pd.Series,
        reference: dict[str, dict[str, float]],
    ) -> list[str]:
        warnings: list[str] = []
        for sensor in ("temperature_c", "humidity_pct", "co2_ppm"):
            value = latest.get(sensor)
            limits = reference.get(sensor, {})
            if value is None or pd.isna(value):
                warnings.append(
                    f"{sensor} is missing in the latest rolling hourly aggregate"
                )
                continue
            if limits and (
                float(value) < float(limits["p01"])
                or float(value) > float(limits["p99"])
            ):
                warnings.append(
                    f"{sensor}={float(value):.2f} is outside the historical "
                    f"1st–99th percentile range ({float(limits['p01']):.2f} to "
                    f"{float(limits['p99']):.2f})"
                )
        return warnings

    @staticmethod
    def _weight_domain_shift(
        data: pd.DataFrame,
        reference: dict[str, dict[str, float]],
    ) -> list[str]:
        if len(data) < 25:
            return []
        weight = pd.to_numeric(data["weight_kg"], errors="coerce")
        previous = weight.shift(24)
        change = ((weight - previous) / previous.abs().clip(lower=1.0)) * 100.0
        value = change.iloc[-1]
        limits = reference.get("weight_change_pct_24h", {})
        if pd.isna(value) or not limits:
            return []
        if float(value) < float(limits["p01"]) or float(value) > float(limits["p99"]):
            return [
                (
                    f"24-hour relative weight change={float(value):.2f}% is outside "
                    f"the historical 1st–99th percentile range "
                    f"({float(limits['p01']):.2f}% to "
                    f"{float(limits['p99']):.2f}%)."
                )
            ]
        return []

    @staticmethod
    def _prediction_matrix(values: Any, horizon: int) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 1:
            matrix = np.repeat(matrix[:, None], horizon, axis=1)
        if matrix.shape[1] != horizon:
            raise ModelNotReadyError(
                f"The saved model returned {matrix.shape[1]} horizons; "
                f"expected {horizon}."
            )
        return np.clip(matrix, 1.0, 100.0)

    @staticmethod
    def _clean_history_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        history = frame.tail(168).copy()
        history["timestamp"] = history["timestamp"].map(
            lambda value: pd.Timestamp(value).isoformat()
        )
        records: list[dict[str, Any]] = []
        for row in history.to_dict(orient="records"):
            cleaned: dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, (np.integer,)):
                    cleaned[key] = int(value)
                elif isinstance(value, (np.floating, float)):
                    cleaned[key] = float(value) if np.isfinite(value) else None
                else:
                    cleaned[key] = value
            records.append(cleaned)
        return records

    def predict_hourly_history(self, hourly: pd.DataFrame) -> dict[str, Any]:
        bundle = self._load_bundle()
        horizon = int(bundle["horizon_hours"])
        data = normalise_historical(hourly)
        if data.empty:
            raise ValueError("No valid hourly IoT data are available for prediction")
        if data["hive_id"].nunique() != 1:
            raise ValueError("Live prediction accepts one hive/device at a time")

        data = data.sort_values(["hive_id", "timestamp"]).reset_index(drop=True)
        data[list(SENSORS)] = data.groupby("hive_id", sort=False)[
            list(SENSORS)
        ].ffill(limit=2)
        data = data.dropna(subset=list(SENSORS)).reset_index(drop=True)
        if data.empty:
            raise ValueError("No complete hourly sensor observation is available")

        feature_frame = build_feature_frame(data).reindex(
            columns=bundle["feature_columns"]
        )
        prediction_matrix = self._prediction_matrix(
            bundle["model"].predict(feature_frame),
            horizon,
        )

        score_config = BroodHealthScoreConfig.from_dict(bundle["score_config"])
        condition = compute_condition_history(data, score_config=score_config)
        result = condition[
            [
                "hive_id",
                "timestamp",
                *SENSORS,
                "temperature_component",
                "humidity_component",
                "co2_component",
                "weight_component",
                "condition_score",
                "condition_level",
                "bhsi",
                "stability_level",
                "rod_points_per_hour",
                "trend_label",
            ]
        ].copy()

        current_scores = result["condition_score"].to_numpy(dtype=float)
        forecast_bhsi_values = forecast_bhsi(
            current_scores,
            prediction_matrix,
            reference=bundle["forecast_stability_reference"],
        )
        forecast_rod_values = forecast_rod(current_scores, prediction_matrix)

        result["exact_forecast_score"] = prediction_matrix[:, -1]
        result["safety_minimum_score"] = prediction_matrix.min(axis=1)
        result["exact_forecast_level"] = result["exact_forecast_score"].map(
            classify_health_level
        )
        result["safety_minimum_level"] = result["safety_minimum_score"].map(
            classify_health_level
        )
        result["exact_forecast_change_points"] = (
            result["exact_forecast_score"] - result["condition_score"]
        )
        result["exact_forecast_drop_points"] = (
            result["condition_score"] - result["exact_forecast_score"]
        ).clip(lower=0.0)
        result["safety_drop_points"] = (
            result["condition_score"] - result["safety_minimum_score"]
        ).clip(lower=0.0)
        result["forecast_bhsi"] = forecast_bhsi_values
        result["forecast_stability_level"] = result["forecast_bhsi"].map(
            classify_stability
        )
        result["forecast_rod_points_per_hour"] = forecast_rod_values
        result["forecast_trend_label"] = result[
            "forecast_rod_points_per_hour"
        ].map(classify_trend)

        latest = result.iloc[-1]
        latest_features = feature_frame.iloc[-1]
        domain_shift = self._environment_domain_shift(
            latest,
            bundle["training_sensor_reference"],
        )
        domain_shift.extend(
            self._weight_domain_shift(data, bundle["training_sensor_reference"])
        )

        history_sufficient = len(data) >= 72
        warning = build_warning_payload(
            exact_forecast_score=float(latest["exact_forecast_score"]),
            safety_minimum_score=float(latest["safety_minimum_score"]),
            current_condition_score=float(latest["condition_score"]),
            forecast_bhsi=float(latest["forecast_bhsi"]),
            forecast_rod_points_per_hour=float(
                latest["forecast_rod_points_per_hour"]
            ),
            exact_forecast_drop_points=float(
                latest["exact_forecast_drop_points"]
            ),
            safety_drop_points=float(latest["safety_drop_points"]),
            domain_shift_warnings=domain_shift,
            history_sufficient=history_sufficient,
            current_temperature_c=float(latest["temperature_c"]),
            current_humidity_pct=float(latest["humidity_pct"]),
            current_co2_ppm=float(latest["co2_ppm"]),
            weight_change_pct_24h=latest_features.get("weight_change_pct_24h"),
            weight_component=float(latest["weight_component"]),
        )

        latest_timestamp = pd.Timestamp(latest["timestamp"])
        if latest_timestamp.tzinfo is None:
            latest_timestamp = latest_timestamp.tz_localize("UTC")
        forecast_timestamp = latest_timestamp + pd.Timedelta(hours=horizon)
        now = pd.Timestamp.now(tz="UTC")
        freshness_minutes = max(
            0.0,
            float((now - latest_timestamp).total_seconds() / 60.0),
        )

        latest_trajectory = prediction_matrix[-1]
        trajectory = [
            {
                "offset_minutes": hour * 60,
                "horizon_hours": hour,
                "forecast_timestamp": (
                    latest_timestamp + pd.Timedelta(hours=hour)
                ).isoformat(),
                "score": float(latest_trajectory[hour - 1]),
                "level": classify_health_level(latest_trajectory[hour - 1]),
                "is_native_model_point": True,
                "value_kind": "native_hourly_model_output",
            }
            for hour in range(1, horizon + 1)
        ]
        display_trajectory = interpolate_forecast_trajectory(
            current_score=float(latest["condition_score"]),
            hourly_scores=latest_trajectory,
            anchor_timestamp=latest_timestamp,
            resolution_minutes=10,
        )
        for point in display_trajectory:
            point["level"] = classify_health_level(point["score"])

        interval = bundle["prediction_interval_absolute_error"]
        half_width_80 = float(interval["80_percent"])
        half_width_90 = float(interval["90_percent"])
        exact_value = float(latest["exact_forecast_score"])

        optional_context: dict[str, float | None] = {}
        for column in (
            "external_temp",
            "external_humidity",
            "battery_voltage",
            "raw_reading_count",
        ):
            if column in data.columns:
                value = data.iloc[-1][column]
                optional_context[column] = (
                    None if pd.isna(value) else float(value)
                )

        context_warnings: list[str] = []
        raw_weight_reference = bundle["training_sensor_reference"].get(
            "weight_kg",
            {},
        )
        if raw_weight_reference:
            live_weight = float(latest["weight_kg"])
            if (
                live_weight
                < float(raw_weight_reference.get("p01", live_weight))
                or live_weight
                > float(raw_weight_reference.get("p99", live_weight))
            ):
                context_warnings.append(
                    "Absolute live hive weight differs from the historical "
                    "hive-weight range. The forecast does not use absolute weight; "
                    "it uses relative weight change and stability."
                )

        return {
            "version": "6.0",
            "device_id": str(latest["hive_id"]),
            "latest_timestamp": latest_timestamp.isoformat(),
            "live_latest_timestamp": latest_timestamp.isoformat(),
            "forecast_timestamp": forecast_timestamp.isoformat(),
            "data_freshness_minutes": freshness_minutes,
            "hourly_rows": len(data),
            "raw_readings_in_latest_hour": optional_context.get(
                "raw_reading_count"
            ),
            "feature_completeness_percentage": self._feature_completeness(
                latest_features
            ),
            "minimum_recommended_history_hours": 72,
            "history_sufficiency": (
                "good" if history_sufficient else "limited"
            ),
            "model": self.model_info(),
            "latest_sensors": {
                sensor: float(latest[sensor]) for sensor in SENSORS
            },
            "latest_raw_sensors": {
                sensor: float(latest[sensor]) for sensor in SENSORS
            },
            "score_components": {
                "temperature": float(latest["temperature_component"]),
                "humidity": float(latest["humidity_component"]),
                "co2": float(latest["co2_component"]),
                "weight_stability": float(latest["weight_component"]),
            },
            "context": optional_context,
            "prediction": {
                "exact_score": exact_value,
                "exact_level": str(latest["exact_forecast_level"]),
                "forecast_anchor_timestamp": latest_timestamp.isoformat(),
                "exact_forecast_timestamp": forecast_timestamp.isoformat(),
                "safety_minimum_score": float(
                    latest["safety_minimum_score"]
                ),
                "safety_minimum_level": str(
                    latest["safety_minimum_level"]
                ),
                "exact_change_points": float(
                    latest["exact_forecast_change_points"]
                ),
                "exact_drop_points": float(
                    latest["exact_forecast_drop_points"]
                ),
                "safety_drop_points": float(latest["safety_drop_points"]),
                "horizon_hours": horizon,
                "native_model_resolution_minutes": 60,
                "display_resolution_minutes": 10,
                "rolling_target_time": True,
                "trajectory": trajectory,
                "display_trajectory": display_trajectory,
                "prediction_interval_80": [
                    float(max(1.0, exact_value - half_width_80)),
                    float(min(100.0, exact_value + half_width_80)),
                ],
                "prediction_interval_90": [
                    float(max(1.0, exact_value - half_width_90)),
                    float(min(100.0, exact_value + half_width_90)),
                ],
                "primary_interpretation": (
                    f"Predicted Brood Health Score exactly {horizon} hours "
                    "after the latest rolling IoT anchor."
                ),
                "secondary_interpretation": (
                    "The safety minimum is the lowest native hourly model value "
                    "inside the predicted 1–6 hour trajectory."
                ),
                "display_interpolation_note": (
                    "Ten-minute trajectory points are linear display interpolation "
                    "between native hourly model outputs. They do not represent a "
                    "separately trained ten-minute model."
                ),
            },
            "forecast_indicators": {
                "bhsi": float(latest["forecast_bhsi"]),
                "stability_level": str(
                    latest["forecast_stability_level"]
                ),
                "rod_points_per_hour": float(
                    latest["forecast_rod_points_per_hour"]
                ),
                "trend_label": str(latest["forecast_trend_label"]),
                "window_start_timestamp": latest_timestamp.isoformat(),
                "window_end_timestamp": forecast_timestamp.isoformat(),
                "source": "current score plus predicted +1 to +6 hour scores",
                "bhsi_definition": bundle[
                    "forecast_indicator_definition"
                ]["bhsi"],
                "rod_definition": bundle[
                    "forecast_indicator_definition"
                ]["rod"],
            },
            "current_condition": {
                "score": float(latest["condition_score"]),
                "level": str(latest["condition_level"]),
                "bhsi": float(latest["bhsi"]),
                "stability_level": str(latest["stability_level"]),
                "rod_points_per_hour": float(
                    latest["rod_points_per_hour"]
                ),
                "trend_label": str(latest["trend_label"]),
                "score_window": (
                    "rolling one-hour median ending at the latest IoT reading"
                ),
            },
            "deployment": {
                "aggregation": (
                    "rolling one-hour median aligned to latest reading"
                ),
                "forecast_anchor_updates_with_each_reading": True,
                "native_model_resolution_minutes": 60,
                "dashboard_trajectory_resolution_minutes": 10,
                "ten_minute_values_are_interpolated": True,
                "external_weather_role": (
                    "display and domain context only; excluded from the trained "
                    "model because the common historical dataset has no equivalent "
                    "external variables"
                ),
            },
            "domain_shift_warnings": domain_shift,
            "context_warnings": context_warnings,
            "warning": warning,
            "history": self._clean_history_records(result),
            "score_definition": {
                "range": "1–100",
                "levels": [
                    "Critical",
                    "Poor",
                    "Good",
                    "Excellent",
                ],
                "is_direct_biological_measurement": False,
                "weight_strategy": "relative_change_and_stability_only",
            },
            "disclaimer": (
                "This is a sensor-based decision-support output. Confirm Poor "
                "or Critical warnings through physical brood inspection. Rolling "
                "ten-minute forecast timestamps increase responsiveness but do not "
                "by themselves prove higher accuracy; field validation is required."
            ),
        }

    def predict_raw_iot(
        self,
        raw: pd.DataFrame,
        *,
        weight_scale_factor: float = 1.0,
        weight_offset_kg: float = 0.0,
    ) -> dict[str, Any]:
        mapped = map_iot_frame(
            raw,
            weight_scale_factor=weight_scale_factor,
            weight_offset_kg=weight_offset_kg,
        )
        if mapped.empty:
            raise ValueError("No valid live IoT readings are available")

        mapped = mapped.sort_values(["hive_id", "timestamp"]).reset_index(
            drop=True
        )
        if mapped["hive_id"].nunique() != 1:
            raise ValueError("Live prediction accepts one hive/device at a time")

        latest_raw = mapped.iloc[-1]
        latest_raw_timestamp = pd.Timestamp(latest_raw["timestamp"])
        if latest_raw_timestamp.tzinfo is None:
            latest_raw_timestamp = latest_raw_timestamp.tz_localize("UTC")

        diffs = (
            mapped["timestamp"]
            .sort_values()
            .drop_duplicates()
            .diff()
            .dt.total_seconds()
            .div(60.0)
        )
        positive_diffs = diffs[diffs > 0]
        reading_interval_minutes = (
            float(positive_diffs.tail(200).median())
            if not positive_diffs.empty
            else None
        )

        hourly = aggregate_live_hourly(
            mapped,
            anchor_to_latest=True,
        )
        result = self.predict_hourly_history(hourly)

        result["live_latest_timestamp"] = latest_raw_timestamp.isoformat()
        result["latest_timestamp"] = latest_raw_timestamp.isoformat()
        result["data_freshness_minutes"] = max(
            0.0,
            float(
                (
                    pd.Timestamp.now(tz="UTC") - latest_raw_timestamp
                ).total_seconds()
                / 60.0
            ),
        )
        result["reading_interval_minutes"] = reading_interval_minutes

        raw_sensor_payload: dict[str, float | None] = {}
        for sensor in SENSORS:
            value = latest_raw.get(sensor)
            raw_sensor_payload[sensor] = (
                None if value is None or pd.isna(value) else float(value)
            )
        result["latest_raw_sensors"] = raw_sensor_payload

        raw_context = dict(result.get("context") or {})
        for column in (
            "external_temp",
            "external_humidity",
            "battery_voltage",
        ):
            if column in mapped.columns:
                value = latest_raw.get(column)
                raw_context[column] = (
                    None if value is None or pd.isna(value) else float(value)
                )
        result["context"] = raw_context
        result["deployment"]["observed_reading_interval_minutes"] = (
            reading_interval_minutes
        )
        return result
