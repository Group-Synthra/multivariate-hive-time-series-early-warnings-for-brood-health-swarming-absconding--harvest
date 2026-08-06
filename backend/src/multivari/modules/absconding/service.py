from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from multivari.common.schema import HIVE_COLUMN, SENSOR_COLUMNS, TIMESTAMP_COLUMN

from .config import AbscondingSettings
from .data import EXTERNAL_SENSOR_COLUMNS
from .features import build_absconding_features
from .iot import IotSettings, SupabaseIotRepository
from .modeling import positive_probability

ALLOWED_IMAGES = {
    "absconding_class_balance.png",
    "absconding_model_comparison.png",
    "absconding_confusion_matrix.png",
    "absconding_feature_importance.png",
}

COLUMN_ALIASES = {
    "recorded_at": TIMESTAMP_COLUMN,
    "reading_at": TIMESTAMP_COLUMN,
    "datetime": TIMESTAMP_COLUMN,
    "device_id": HIVE_COLUMN,
    "internal_temp": "temperature_c",
    "internal_temperature_c": "temperature_c",
    "temperature": "temperature_c",
    "internal_humidity": "humidity_pct",
    "internal_humidity_pct": "humidity_pct",
    "humidity": "humidity_pct",
    "internal_co2": "co2_ppm",
    "co2": "co2_ppm",
    "total_weight": "weight_kg",
    "hive_weight_kg": "weight_kg",
    "weight": "weight_kg",
    "external_temp": "external_temperature_c",
    "external_temperature_c": "external_temperature_c",
    "external_humidity": "external_humidity_pct",
    "external_humidity_pct": "external_humidity_pct",
}


@dataclass
class AbscondingService:
    backend_root: Path
    _dashboard_cache_signature: tuple[int, int] | None = field(default=None, init=False)
    _dashboard_cache: dict[str, Any] | None = field(default=None, init=False)
    _model_cache_signature: tuple[int, int] | None = field(default=None, init=False)
    _model_cache: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.backend_root = Path(self.backend_root).resolve()
        self.report_directory = self.backend_root / "artifacts" / "reports" / "absconding"
        self.dashboard_path = self.report_directory / "absconding_dashboard.json"
        self.model_path = (
            self.backend_root
            / "artifacts"
            / "models"
            / "absconding"
            / "absconding_model_bundle.joblib"
        )

    def get_dashboard(self) -> dict[str, Any]:
        if not self.dashboard_path.is_file():
            raise FileNotFoundError(
                "Absconding outputs were not found. Run: python scripts/run_absconding_pipeline.py"
            )
        stat = self.dashboard_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if self._dashboard_cache_signature != signature or self._dashboard_cache is None:
            self._dashboard_cache = json.loads(self.dashboard_path.read_text(encoding="utf-8"))
            self._dashboard_cache_signature = signature
        return self._dashboard_cache

    def get_hive(self, hive_id: str) -> dict[str, Any]:
        dashboard = self.get_dashboard()
        details = dashboard.get("hive_details", {})
        if hive_id not in details:
            raise KeyError(hive_id)
        return details[hive_id]

    def image_path(self, filename: str) -> Path:
        if filename not in ALLOWED_IMAGES:
            raise FileNotFoundError(f"Unknown absconding image: {filename}")
        path = (self.report_directory / filename).resolve()
        if path.parent != self.report_directory.resolve() or not path.is_file():
            raise FileNotFoundError(f"Absconding image was not found: {filename}")
        return path

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        readings = payload.get("readings")
        if not isinstance(readings, list) or not readings:
            raise ValueError("Send a non-empty JSON array in the 'readings' field.")

        bundle, settings, hourly, scored = self._score_readings(pd.DataFrame(readings))
        history_sizes = hourly.groupby(HIVE_COLUMN, sort=False).size().to_dict()
        predictions = []
        for hive_id, group in scored.groupby(HIVE_COLUMN, sort=True):
            latest = group.sort_values(TIMESTAMP_COLUMN).iloc[-1]
            predictions.append(
                self._prediction_payload(
                    latest,
                    bundle=bundle,
                    settings=settings,
                    history_rows=int(history_sizes.get(hive_id, len(group))),
                    scored_rows=len(group),
                )
            )

        return {
            "model_name": bundle["model_name"],
            "prediction_horizon_hours": bundle["prediction_horizon_hours"],
            "input_rows": len(readings),
            "hourly_rows": len(hourly),
            "predictions": predictions,
            "warning": (
                "This model uses the separate labelled Absconding dataset. Treat live predictions "
                "as research early warnings until locally labelled Sri Lankan events are available "
                "for calibration and biological validation."
            ),
        }

    def build_live_iot_prediction(self) -> dict[str, Any]:
        iot_settings = IotSettings.from_env()
        repository = SupabaseIotRepository(iot_settings)
        raw, source_metadata = repository.fetch_latest()
        if raw.empty:
            raise ValueError("The IoT database returned no sensor records.")

        latest_raw = raw.iloc[-1]
        latest_timestamp = pd.to_datetime(
            latest_raw.get("recorded_at"), errors="coerce", utc=iot_settings.timestamps_are_utc
        )
        latest_sensors = _raw_sensor_payload(latest_raw)

        try:
            bundle, settings, hourly, scored = self._score_readings(
                raw,
                feature_timezone=iot_settings.feature_timezone,
                timestamps_are_utc=iot_settings.timestamps_are_utc,
            )
        except ValueError as error:
            # The live tab remains useful while the required history is accumulating,
            # but it never invents a model probability.
            if "At least" not in str(error):
                raise
            hourly = self._aggregate_hourly(
                self._normalise_readings(
                    raw,
                    feature_timezone=iot_settings.feature_timezone,
                    timestamps_are_utc=iot_settings.timestamps_are_utc,
                )
            )
            return self._collecting_history_payload(
                raw=raw,
                hourly=hourly,
                latest_timestamp=latest_timestamp,
                latest_sensors=latest_sensors,
                source_metadata=source_metadata,
                iot_settings=iot_settings,
                reason=str(error),
            )

        latest = scored.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN]).iloc[-1]
        prediction = self._prediction_payload(
            latest,
            bundle=bundle,
            settings=settings,
            history_rows=len(hourly),
            scored_rows=len(scored),
        )
        factors = _signal_explanations(latest)
        recommendation = _recommended_action(prediction["risk_level"], prediction["arm"], factors)

        timeline = []
        for _, row in scored.tail(168).iterrows():
            timeline.append(
                {
                    "timestamp": row[TIMESTAMP_COLUMN].isoformat(),
                    "risk_probability": round(float(row["probability"]), 8),
                    "risk_percentage": round(float(row["probability"] * 100), 4),
                    "risk_level": _risk_level(
                        float(row["probability"]),
                        float(row["arm"]),
                        medium=float(bundle["medium_threshold"]),
                        high=float(bundle["alert_threshold"]),
                        arm_threshold=settings.arm_escalation_threshold,
                    ),
                    "arm": round(float(row["arm"]), 8),
                    "arm_per_hour": round(float(row["arm_per_hour"]), 10),
                    "arm_change_percentage_points": round(float(row["arm"] * 100), 4),
                    "arm_trend": _arm_label(float(row["arm"])),
                    "environmental_stress_score": _optional_float(
                        row.get("environmental_stress_score"), 6
                    ),
                    "temperature_c": _optional_float(row.get("temperature_c"), 3),
                    "humidity_pct": _optional_float(row.get("humidity_pct"), 3),
                    "co2_ppm": _optional_float(row.get("co2_ppm"), 3),
                    "weight_kg": _optional_float(row.get("weight_kg"), 3),
                }
            )

        age_minutes = _age_minutes(latest_timestamp)
        freshness = _freshness(age_minutes, iot_settings.interval_minutes)
        latest_sensors.update(_feature_sensor_payload(latest))
        notification = {
            "should_notify": prediction["risk_level"] == "High"
            or (
                prediction["risk_level"] == "Medium"
                and prediction["arm"] >= settings.arm_escalation_threshold
            ),
            "title": (
                "Absconding Early Warning"
                if prediction["risk_level"] != "Low"
                else "Absconding Risk Normal"
            ),
            "message": recommendation,
            "risk_level": prediction["risk_level"],
            "risk_percentage": prediction["risk_percentage"],
            "arm": prediction["arm"],
            "arm_trend": prediction["arm_trend"],
        }

        return {
            "mode": "real_time_iot",
            "status": "ok",
            "data_source": source_metadata,
            "hive_id": prediction["hive_id"],
            "prediction_window": f"next_{settings.prediction_horizon_hours}_hours",
            "sampling_interval_minutes": iot_settings.interval_minutes,
            "records_available": len(raw),
            "hourly_records_available": len(hourly),
            "records_used_for_prediction": len(scored),
            "minimum_history_hours": settings.minimum_history_hours,
            "last_updated": prediction["timestamp"],
            "database_last_reading": (
                latest_timestamp.isoformat() if not pd.isna(latest_timestamp) else None
            ),
            "data_age_minutes": age_minutes,
            "data_freshness_status": freshness,
            "next_expected_reading": (
                (latest_timestamp + pd.Timedelta(minutes=iot_settings.interval_minutes)).isoformat()
                if not pd.isna(latest_timestamp)
                else None
            ),
            "active_model_name": bundle["model_name"],
            "active_model_family": bundle.get("model_family"),
            "risk_probability": prediction["probability"],
            "risk_percentage": prediction["risk_percentage"],
            "current_probability": prediction["current_probability"],
            "current_probability_percent": prediction["current_probability_percent"],
            "previous_probability": prediction["previous_probability"],
            "previous_probability_percent": prediction["previous_probability_percent"],
            "previous_probability_timestamp": prediction["previous_probability_timestamp"],
            "probability_change": prediction["probability_change"],
            "probability_change_percentage_points": prediction[
                "probability_change_percentage_points"
            ],
            "comparison_hours": prediction["comparison_hours"],
            "risk_level": prediction["risk_level"],
            "arm": prediction["arm"],
            "arm_per_hour": prediction["arm_per_hour"],
            "arm_trend": prediction["arm_trend"],
            "thresholds": {
                "medium_probability": round(float(bundle["medium_threshold"]), 8),
                "medium_percentage": round(float(bundle["medium_threshold"]) * 100, 4),
                "high_probability": round(float(bundle["alert_threshold"]), 8),
                "high_percentage": round(float(bundle["alert_threshold"]) * 100, 4),
                "arm_escalation_probability_change": round(
                    float(settings.arm_escalation_threshold), 8
                ),
                "arm_escalation_percentage_points": round(
                    float(settings.arm_escalation_threshold) * 100, 4
                ),
            },
            "notification": notification,
            "recommended_action": recommendation,
            "recommended_actions": _action_checklist(prediction["risk_level"], factors),
            "latest_sensor_readings": latest_sensors,
            "key_factors": factors,
            "timeline": timeline,
            "live_note": (
                "The model is trained offline. The backend polls Supabase every 10 minutes, "
                "applies the same feature pipeline and performs inference only."
            ),
            "evidence_note": (
                "Treat the prediction as an exploratory early-warning aid until more locally "
                "labelled absconding events are available."
            ),
        }

    def _collecting_history_payload(
        self,
        *,
        raw: pd.DataFrame,
        hourly: pd.DataFrame,
        latest_timestamp: pd.Timestamp,
        latest_sensors: dict[str, Any],
        source_metadata: dict[str, Any],
        iot_settings: IotSettings,
        reason: str,
    ) -> dict[str, Any]:
        bundle = self._load_model_bundle()
        required = int(bundle.get("minimum_history_hours", 168))
        available = int(hourly.groupby(HIVE_COLUMN).size().max()) if not hourly.empty else 0
        age_minutes = _age_minutes(latest_timestamp)
        return {
            "mode": "real_time_iot",
            "status": "collecting_history",
            "data_source": source_metadata,
            "hive_id": str(raw.iloc[-1].get("device_id", "unknown")),
            "sampling_interval_minutes": iot_settings.interval_minutes,
            "records_available": len(raw),
            "hourly_records_available": available,
            "minimum_history_hours": required,
            "history_progress_percentage": round(min(100.0, available / required * 100), 2),
            "last_updated": (
                latest_timestamp.isoformat() if not pd.isna(latest_timestamp) else None
            ),
            "data_age_minutes": age_minutes,
            "data_freshness_status": _freshness(age_minutes, iot_settings.interval_minutes),
            "risk_probability": None,
            "risk_percentage": None,
            "risk_level": "Pending",
            "arm": None,
            "arm_trend": "Collecting history",
            "latest_sensor_readings": latest_sensors,
            "key_factors": [],
            "timeline": [],
            "notification": {
                "should_notify": False,
                "title": "Collecting IoT history",
                "message": (
                    f"{available} of {required} hourly observations are available. "
                    "The system will not invent a risk score before the trained feature window is complete."
                ),
            },
            "recommended_action": "Continue collecting sensor readings and verify data quality.",
            "reason": reason,
        }

    def _score_readings(
        self,
        frame: pd.DataFrame,
        *,
        feature_timezone: str | None = None,
        timestamps_are_utc: bool = True,
    ) -> tuple[dict[str, Any], AbscondingSettings, pd.DataFrame, pd.DataFrame]:
        bundle = self._load_model_bundle()
        settings = AbscondingSettings(**_restore_settings(bundle["settings"]))
        normalised = self._normalise_readings(
            frame,
            feature_timezone=feature_timezone,
            timestamps_are_utc=timestamps_are_utc,
        )
        hourly = self._aggregate_hourly(normalised)
        features = build_absconding_features(hourly, settings)
        valid = features.loc[features["has_full_absconding_history"].eq(1)].copy()
        if valid.empty:
            available = int(hourly.groupby(HIVE_COLUMN).size().max()) if not hourly.empty else 0
            raise ValueError(
                f"At least {settings.minimum_history_hours} hourly observations per hive are required; "
                f"the largest supplied history contains {available}."
            )

        for feature in bundle["feature_names"]:
            if feature not in valid:
                valid[feature] = np.nan
        valid["probability"] = positive_probability(
            bundle["estimator"], valid[bundle["feature_names"]]
        )
        valid = valid.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN]).copy()
        grouped = valid.groupby(HIVE_COLUMN, sort=False)
        valid["previous_probability"] = grouped["probability"].shift(settings.arm_change_hours)
        valid["previous_probability_timestamp"] = grouped[TIMESTAMP_COLUMN].shift(
            settings.arm_change_hours
        )
        valid["probability_change"] = valid["probability"] - valid["previous_probability"]
        valid["arm"] = valid["probability_change"].fillna(0.0)
        valid["arm_per_hour"] = valid["arm"] / max(settings.arm_change_hours, 1)
        return bundle, settings, hourly, valid

    def _prediction_payload(
        self,
        latest: pd.Series,
        *,
        bundle: dict[str, Any],
        settings: AbscondingSettings,
        history_rows: int,
        scored_rows: int,
    ) -> dict[str, Any]:
        probability = float(latest["probability"])
        arm = float(latest["arm"])
        previous_probability = _finite_or_none(latest.get("previous_probability"))
        previous_timestamp = _timestamp_or_none(latest.get("previous_probability_timestamp"))
        probability_change = (
            probability - previous_probability if previous_probability is not None else None
        )
        risk_level = _risk_level(
            probability,
            arm,
            medium=float(bundle["medium_threshold"]),
            high=float(bundle["alert_threshold"]),
            arm_threshold=settings.arm_escalation_threshold,
        )
        return {
            "hive_id": str(latest[HIVE_COLUMN]),
            "timestamp": latest[TIMESTAMP_COLUMN].isoformat(),
            "probability": round(probability, 8),
            "risk_percentage": round(probability * 100, 4),
            "current_probability": round(probability, 8),
            "current_probability_percent": round(probability * 100, 4),
            "previous_probability": (
                round(previous_probability, 8) if previous_probability is not None else None
            ),
            "previous_probability_percent": (
                round(previous_probability * 100, 4)
                if previous_probability is not None
                else None
            ),
            "previous_probability_timestamp": previous_timestamp,
            "probability_change": (
                round(probability_change, 8) if probability_change is not None else None
            ),
            "probability_change_percentage_points": (
                round(probability_change * 100, 4)
                if probability_change is not None
                else None
            ),
            "comparison_hours": int(settings.arm_change_hours),
            "risk_level": risk_level,
            "arm": round(arm, 8),
            "arm_per_hour": round(float(latest["arm_per_hour"]), 10),
            "arm_trend": _arm_label(arm),
            "hourly_history_supplied": history_rows,
            "scored_observations": scored_rows,
        }

    def _load_model_bundle(self) -> dict[str, Any]:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                "Absconding model was not found. Run: python scripts/run_absconding_pipeline.py"
            )
        stat = self.model_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if self._model_cache_signature != signature or self._model_cache is None:
            self._model_cache = joblib.load(self.model_path)
            self._model_cache_signature = signature
        return self._model_cache

    @staticmethod
    def _normalise_readings(
        frame: pd.DataFrame,
        *,
        feature_timezone: str | None = None,
        timestamps_are_utc: bool = True,
    ) -> pd.DataFrame:
        frame = frame.rename(
            columns={column: COLUMN_ALIASES.get(column, column) for column in frame}
        )
        required = [TIMESTAMP_COLUMN, HIVE_COLUMN, *SENSOR_COLUMNS]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"Missing live inference columns: {missing}")
        optional = [column for column in EXTERNAL_SENSOR_COLUMNS if column in frame]
        frame = frame[[*required, *optional]].copy()
        timestamp = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce", utc=timestamps_are_utc)
        if timestamp.isna().any():
            raise ValueError("One or more live timestamps are invalid.")
        if timestamp.dt.tz is None:
            timestamp = timestamp.dt.tz_localize("UTC")
        if feature_timezone:
            timestamp = timestamp.dt.tz_convert(feature_timezone)
        else:
            timestamp = timestamp.dt.tz_convert("UTC")
        frame[TIMESTAMP_COLUMN] = timestamp.dt.tz_localize(None)
        frame[HIVE_COLUMN] = frame[HIVE_COLUMN].astype("string").str.strip()
        for sensor in (*SENSOR_COLUMNS, *optional):
            frame[sensor] = pd.to_numeric(frame[sensor], errors="coerce")
        return (
            frame.dropna(subset=[HIVE_COLUMN])
            .sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN])
            .drop_duplicates([HIVE_COLUMN, TIMESTAMP_COLUMN], keep="last")
        )

    @staticmethod
    def _aggregate_hourly(frame: pd.DataFrame) -> pd.DataFrame:
        hourly = frame.copy()
        hourly[TIMESTAMP_COLUMN] = hourly[TIMESTAMP_COLUMN].dt.floor("h")
        value_columns = [
            *SENSOR_COLUMNS,
            *[column for column in EXTERNAL_SENSOR_COLUMNS if column in hourly],
        ]
        return (
            hourly.groupby([HIVE_COLUMN, TIMESTAMP_COLUMN], as_index=False)[value_columns]
            .mean()
            .sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN])
            .reset_index(drop=True)
        )


def _restore_settings(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    for key in (
        "lags_hours",
        "change_hours",
        "rolling_windows_hours",
        "rolling_statistics",
        "model_candidates",
    ):
        if key in result:
            result[key] = tuple(result[key])
    return result


def _risk_level(
    probability: float,
    arm: float,
    *,
    medium: float,
    high: float,
    arm_threshold: float,
) -> str:
    if probability >= high:
        return "High"
    if probability >= medium or arm >= arm_threshold:
        return "Medium"
    return "Low"


def _arm_label(arm: float) -> str:
    # ARM is stored as a probability change over the configured comparison
    # window. 0.00005 therefore equals 0.005 percentage points. These labels
    # improve display sensitivity without changing the actual model score or
    # the alert threshold used by the decision engine.
    if arm >= 0.05:
        return "Rapidly Increasing"
    if arm >= 0.01:
        return "Increasing"
    if arm >= 0.00005:
        return "Slightly Increasing"
    if arm <= -0.05:
        return "Rapidly Decreasing"
    if arm <= -0.01:
        return "Decreasing"
    if arm <= -0.00005:
        return "Slightly Decreasing"
    return "Stable"


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _timestamp_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.isoformat()


def _signal_explanations(row: pd.Series) -> list[dict[str, Any]]:
    candidates = [
        (
            "Weight decline",
            max(0.0, -_number(row.get("weight_kg_change_72h"))),
            f"72-hour weight change: {_optional_float(row.get('weight_kg_change_72h'), 3)} kg",
        ),
        (
            "CO₂ buildup",
            max(0.0, _number(row.get("co2_ppm_z_72h"))),
            f"CO₂ trailing deviation: {_optional_float(row.get('co2_ppm_z_72h'), 3)} SD",
        ),
        (
            "Temperature instability",
            abs(_number(row.get("temperature_c_z_72h"))),
            f"Temperature trailing deviation: {_optional_float(row.get('temperature_c_z_72h'), 3)} SD",
        ),
        (
            "Humidity deviation",
            abs(_number(row.get("humidity_pct_z_72h"))),
            f"Humidity trailing deviation: {_optional_float(row.get('humidity_pct_z_72h'), 3)} SD",
        ),
        (
            "Combined environmental stress",
            _number(row.get("environmental_stress_score")),
            f"Environmental stress score: {_optional_float(row.get('environmental_stress_score'), 3)}",
        ),
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [
        {
            "factor": factor,
            "signal_strength": round(float(max(score, 0.0)), 4),
            "detail": detail,
        }
        for factor, score, detail in candidates[:4]
    ]


def _recommended_action(
    risk_level: str,
    arm: float,
    factors: list[dict[str, Any]],
) -> str:
    factor_names = ", ".join(item["factor"] for item in factors[:3]) or "combined stress"
    if risk_level == "High":
        return (
            "Inspect the hive within the next 12–24 hours. Check queen status, food stores, "
            f"ventilation, pests and colony activity. Main signals: {factor_names}."
        )
    if risk_level == "Medium" and arm > 0:
        return (
            "Risk is increasing. Monitor the next readings closely and prepare a physical "
            f"inspection. Main signals: {factor_names}."
        )
    if risk_level == "Medium":
        return "Monitor closely and inspect the hive if the warning persists."
    return "Continue routine monitoring and maintain the 10-minute IoT data stream."


def _action_checklist(
    risk_level: str,
    factors: list[dict[str, Any]],
) -> list[str]:
    actions = ["Verify sensor freshness and inspect recent hive activity."]
    factor_names = {item["factor"] for item in factors}
    if "CO₂ buildup" in factor_names:
        actions.append("Check hive ventilation and entrance obstruction.")
    if "Weight decline" in factor_names:
        actions.append("Inspect food stores, colony strength and possible bee movement.")
    if "Temperature instability" in factor_names:
        actions.append("Check shading, insulation and internal temperature disturbance.")
    if "Humidity deviation" in factor_names:
        actions.append("Inspect moisture, condensation and airflow conditions.")
    if risk_level == "High":
        actions.insert(0, "Perform an urgent physical hive inspection within 12–24 hours.")
    return actions


def _raw_sensor_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "temperature_c": _optional_float(row.get("internal_temp"), 3),
        "humidity_pct": _optional_float(row.get("internal_humidity"), 3),
        "co2_ppm": _optional_float(row.get("internal_co2"), 3),
        "weight_kg": _optional_float(row.get("total_weight"), 3),
        "external_temperature_c": _optional_float(row.get("external_temp"), 3),
        "external_humidity_pct": _optional_float(row.get("external_humidity"), 3),
        "battery_voltage": _optional_float(row.get("battery_voltage"), 3),
    }


def _feature_sensor_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "environmental_stress_score": _optional_float(row.get("environmental_stress_score"), 6),
        "stress_trend_24h": _optional_float(row.get("stress_trend_24h"), 6),
        "weight_change_1h": _optional_float(row.get("weight_kg_change_1h"), 3),
        "weight_change_6h": _optional_float(row.get("weight_kg_change_6h"), 3),
        "weight_change_24h": _optional_float(row.get("weight_kg_change_24h"), 3),
        "weight_change_72h": _optional_float(row.get("weight_kg_change_72h"), 3),
        "co2_change_6h": _optional_float(row.get("co2_ppm_change_6h"), 3),
        "co2_change_24h": _optional_float(row.get("co2_ppm_change_24h"), 3),
        "temp_deviation_from_35": _optional_float(row.get("temperature_deviation_from_35"), 3),
        "humidity_deviation_from_optimal": _optional_float(
            row.get("humidity_deviation_from_optimal"), 3
        ),
        "external_temperature_c": _optional_float(row.get("external_temperature_c"), 3),
        "external_humidity_pct": _optional_float(row.get("external_humidity_pct"), 3),
        "internal_external_temperature_difference": _optional_float(
            row.get("internal_external_temperature_difference"), 3
        ),
        "internal_external_humidity_difference": _optional_float(
            row.get("internal_external_humidity_difference"), 3
        ),
        "co2_high_flag": int(_number(row.get("co2_high_flag"))),
        "rapid_weight_loss_flag": int(_number(row.get("rapid_weight_loss_flag"))),
        "sustained_weight_loss_24h": int(_number(row.get("sustained_weight_loss_24h"))),
        "sustained_weight_loss_72h": int(_number(row.get("sustained_weight_loss_72h"))),
    }


def _optional_float(value: Any, digits: int = 4) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _number(value: Any) -> float:
    result = _optional_float(value, 8)
    return float(result) if result is not None else 0.0


def _age_minutes(timestamp: pd.Timestamp) -> float | None:
    if pd.isna(timestamp):
        return None
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return round((pd.Timestamp(datetime.now(UTC)) - stamp).total_seconds() / 60.0, 1)


def _freshness(age_minutes: float | None, interval_minutes: int) -> str:
    if age_minutes is None:
        return "Unknown"
    if age_minutes <= interval_minutes * 2:
        return "Fresh"
    if age_minutes <= interval_minutes * 6:
        return "Stale"
    return "Delayed"