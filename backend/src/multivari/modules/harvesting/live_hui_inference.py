from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from multivari.iot.postgres_repository import PostgresSensorSettings
from multivari.modules.harvesting.classifier_derived_hui import (
    CURRENT_CLASS_COLUMN,
    CURRENT_HUI_COLUMN,
    HIVE_COLUMN,
    SPLIT_COLUMN,
    TIMESTAMP_COLUMN,
    add_hui_history_features,
    assign_harvest_readiness_class,
    probability_to_hui,
)
from multivari.modules.harvesting.hourly_gap_interpolation import interpolate_bounded_hourly_gaps
from multivari.modules.harvesting.reviewed_features import build_reviewed_feature_dataset

LIVE_TARGET_COLUMN = "_live_target_placeholder"
HORIZONS = (24, 48, 72)
REQUIRED_SENSOR_COLUMNS = (
    "weight_kg",
    "temperature_c",
    "humidity_pct",
    "co2_ppm",
)


class LiveHuiInferenceError(RuntimeError):
    """Base error for live HUI inference."""


class LiveHuiArtifactError(LiveHuiInferenceError):
    """Raised when a trained artifact required for inference is missing."""


class InsufficientLiveHistoryError(LiveHuiInferenceError):
    """Raised when no hive has enough contiguous hourly history."""

    def __init__(self, message: str, *, diagnostics: list[dict[str, Any]]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise LiveHuiArtifactError(f"Required live-inference artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _model_label(value: str) -> str:
    lowered = value.lower()
    if lowered == "lightgbm":
        return "LightGBM"
    if lowered == "xgboost":
        return "XGBoost"
    return value.replace("_", " ").title()


@dataclass(frozen=True)
class LiveHuiArtifactSettings:
    classifier_model_path: Path
    classifier_features_path: Path
    calibrator_path: Path
    harvesting_config_path: Path
    calibration_gate_path: Path
    future_hui_gate_path: Path
    future_model_directory: Path
    stale_after_minutes: int
    series_rows_per_hive: int
    training_feature_dataset_path: Path | None = None

    @classmethod
    def from_env(cls, *, backend_root: Path) -> LiveHuiArtifactSettings:
        root = Path(backend_root).resolve()
        return cls(
            classifier_model_path=_resolve(
                root,
                os.getenv(
                    "IOT_CLASSIFIER_MODEL_PATH",
                    "artifacts/models/harvesting/research_v2/selected_model.joblib",
                ),
            ),
            classifier_features_path=_resolve(
                root,
                os.getenv(
                    "IOT_CLASSIFIER_FEATURES_PATH",
                    "artifacts/models/harvesting/research_v2/selected_feature_columns.json",
                ),
            ),
            calibrator_path=_resolve(
                root,
                os.getenv(
                    "IOT_CALIBRATOR_PATH",
                    "artifacts/models/harvesting/probability_calibration/selected_probability_calibrator.joblib",
                ),
            ),
            harvesting_config_path=_resolve(
                root,
                os.getenv("IOT_HARVESTING_CONFIG_PATH", "config/harvesting.yaml"),
            ),
            calibration_gate_path=_resolve(
                root,
                os.getenv(
                    "IOT_CALIBRATION_GATE_PATH",
                    "artifacts/reports/harvesting/reviewed/probability_calibration/probability_calibration_gate.json",
                ),
            ),
            future_hui_gate_path=_resolve(
                root,
                os.getenv(
                    "IOT_FUTURE_HUI_GATE_PATH",
                    "artifacts/reports/harvesting/reviewed/classifier_derived_hui/future_hui_regression_gate.json",
                ),
            ),
            future_model_directory=_resolve(
                root,
                os.getenv(
                    "IOT_FUTURE_HUI_MODEL_DIRECTORY",
                    "artifacts/models/harvesting/classifier_derived_hui_regression",
                ),
            ),
            stale_after_minutes=int(os.getenv("IOT_STALE_AFTER_MINUTES", "30")),
            series_rows_per_hive=int(os.getenv("IOT_SERIES_ROWS_PER_HIVE", "168")),
            training_feature_dataset_path=_resolve(
                root,
                os.getenv(
                    "IOT_TRAINING_FEATURE_DATASET_PATH",
                    "data/processed/harvest_reviewed_feature_dataset.parquet",
                ),
            ),
        )


@dataclass
class FutureModelArtifact:
    horizon: int
    model_name: str
    feature_columns: list[str]
    estimator: Any


class LiveHuiInferenceEngine:
    """Apply the frozen classifier, HUI transformation and future-HUI models."""

    def __init__(
        self,
        *,
        backend_root: str | Path,
        sensor_settings: PostgresSensorSettings,
        artifact_settings: LiveHuiArtifactSettings | None = None,
    ) -> None:
        self.backend_root = Path(backend_root).resolve()
        self.sensor_settings = sensor_settings
        self.artifact_settings = artifact_settings or LiveHuiArtifactSettings.from_env(
            backend_root=self.backend_root
        )

        settings = self.artifact_settings
        for path in (
            settings.classifier_model_path,
            settings.classifier_features_path,
            settings.calibrator_path,
            settings.harvesting_config_path,
            settings.calibration_gate_path,
            settings.future_hui_gate_path,
        ):
            if not path.exists():
                raise LiveHuiArtifactError(f"Required live-inference artifact is missing: {path}")

        self.classifier = joblib.load(settings.classifier_model_path)
        feature_payload = _read_json(settings.classifier_features_path)
        self.classifier_features = [str(value) for value in feature_payload["features"]]
        self.calibrator = joblib.load(settings.calibrator_path)
        self.calibration_gate = _read_json(settings.calibration_gate_path)
        self.future_hui_gate = _read_json(settings.future_hui_gate_path)

        config = yaml.safe_load(settings.harvesting_config_path.read_text(encoding="utf-8"))
        self.reviewed_feature_config = dict(config["reviewed_features"])
        hui_config = dict(config["classifier_derived_hui"])
        self.class_config = {str(key): float(value) for key, value in hui_config["classes"].items()}
        self.probability_anchors = [
            float(anchor["calibrated_score"]) for anchor in hui_config["hui_anchors"]
        ]
        self.hui_anchors = [float(anchor["hui"]) for anchor in hui_config["hui_anchors"]]
        self.future_models = self._load_future_models(settings.future_model_directory)
        self.training_sensor_profile = self._load_training_sensor_profile(
            settings.training_feature_dataset_path
        )

    @staticmethod
    def _load_training_sensor_profile(
        dataset_path: Path | None,
    ) -> dict[str, dict[str, float]]:
        if dataset_path is None or not dataset_path.exists():
            return {}

        column_map = {
            "weight_kg": "weight_kg_current",
            "internal_temperature_c": "temperature_c_current",
            "co2_ppm": "co2_ppm_current",
        }
        requested = [SPLIT_COLUMN, *column_map.values()]
        try:
            frame = pd.read_parquet(dataset_path, columns=requested)
        except (FileNotFoundError, KeyError, ValueError):
            return {}

        training = frame.loc[frame[SPLIT_COLUMN].eq("train")].copy()
        profile: dict[str, dict[str, float]] = {}
        for live_key, column in column_map.items():
            values = pd.to_numeric(training[column], errors="coerce").dropna()
            if values.empty:
                continue
            profile[live_key] = {
                "q01": float(values.quantile(0.01)),
                "q99": float(values.quantile(0.99)),
            }
        return profile

    @staticmethod
    def _load_future_models(model_directory: Path) -> dict[int, FutureModelArtifact]:
        artifacts: dict[int, FutureModelArtifact] = {}
        for horizon in HORIZONS:
            model_path = (
                model_directory / f"selected_classifier_derived_hui_regressor_{horizon}h.joblib"
            )
            metadata_path = model_path.with_suffix(".json")
            if not model_path.exists() or not metadata_path.exists():
                raise LiveHuiArtifactError(
                    f"Missing selected future-HUI artifact for {horizon}h: {model_path}"
                )
            metadata = _read_json(metadata_path)
            artifacts[horizon] = FutureModelArtifact(
                horizon=horizon,
                model_name=str(metadata["selected_model"]),
                feature_columns=[str(value) for value in metadata["feature_columns"]],
                estimator=joblib.load(model_path),
            )
        return artifacts

    def _parse_source_timestamp(self, values: pd.Series) -> pd.Series:
        if self.sensor_settings.timestamps_are_utc:
            timestamp = pd.to_datetime(values, errors="coerce", utc=True)
        else:
            timestamp = pd.to_datetime(values, errors="coerce")
            if timestamp.dt.tz is None:
                timestamp = timestamp.dt.tz_localize(
                    self.sensor_settings.feature_timezone,
                    ambiguous="NaT",
                    nonexistent="shift_forward",
                )
            timestamp = timestamp.dt.tz_convert("UTC")
        return timestamp

    def prepare_hourly_history(
        self,
        raw: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
        required = {
            "source_hive_id",
            "source_timestamp",
            "internal_temperature",
            "internal_humidity",
            "internal_co2",
            "total_weight",
        }
        missing = sorted(required.difference(raw.columns))
        if missing:
            raise LiveHuiInferenceError(f"Live PostgreSQL rows are missing columns: {missing}")
        if raw.empty:
            raise InsufficientLiveHistoryError(
                "No live sensor rows were returned from PostgreSQL.",
                diagnostics=[],
            )

        frame = raw.copy()
        frame["source_timestamp_utc"] = self._parse_source_timestamp(frame["source_timestamp"])
        frame = frame.loc[frame["source_timestamp_utc"].notna()].copy()
        frame[HIVE_COLUMN] = frame["source_hive_id"].astype("string").str.strip()
        frame = frame.loc[frame[HIVE_COLUMN].notna() & frame[HIVE_COLUMN].ne("")]

        numeric_mapping = {
            "internal_temperature": ("temperature_c", self.sensor_settings.temperature_scale),
            "internal_humidity": ("humidity_pct", self.sensor_settings.humidity_scale),
            "internal_co2": ("co2_ppm", self.sensor_settings.co2_scale),
            "total_weight": ("weight_kg", self.sensor_settings.weight_scale),
            "external_temperature": ("external_temperature_c", 1.0),
            "external_humidity": ("external_humidity_pct", 1.0),
            "battery_voltage": ("battery_voltage", 1.0),
        }
        for source, (target, scale) in numeric_mapping.items():
            if source in frame.columns:
                frame[target] = pd.to_numeric(frame[source], errors="coerce") * scale
            else:
                frame[target] = np.nan

        frame["timestamp_feature"] = (
            frame["source_timestamp_utc"]
            .dt.tz_convert(self.sensor_settings.feature_timezone)
            .dt.floor("h")
        )
        frame = frame.sort_values([HIVE_COLUMN, "source_timestamp_utc"])
        frame = frame.drop_duplicates(subset=[HIVE_COLUMN, "source_timestamp_utc"], keep="last")

        latest_raw = frame.groupby(HIVE_COLUMN, group_keys=False, sort=True).tail(1).copy()

        hourly_columns = [
            "weight_kg",
            "temperature_c",
            "humidity_pct",
            "co2_ppm",
            "external_temperature_c",
            "external_humidity_pct",
            "battery_voltage",
        ]
        grouped = frame.groupby(
            [HIVE_COLUMN, "timestamp_feature"],
            observed=True,
        )
        hourly = (
            grouped[hourly_columns]
            .median()
            .reset_index()
            .rename(columns={"timestamp_feature": TIMESTAMP_COLUMN})
        )
        hourly["readings_in_hour"] = grouped.size().to_numpy()

        required_counts = grouped[list(REQUIRED_SENSOR_COLUMNS)].count().reset_index()
        required_counts = required_counts.rename(
            columns={column: f"{column}_reading_count" for column in REQUIRED_SENSOR_COLUMNS}
        )
        hourly = hourly.merge(
            required_counts,
            left_on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
            right_on=[HIVE_COLUMN, "timestamp_feature"],
            how="left",
            validate="one_to_one",
        ).drop(columns=["timestamp_feature"])

        minimum_readings = self.sensor_settings.minimum_readings_per_hour
        for column in REQUIRED_SENSOR_COLUMNS:
            count_column = f"{column}_reading_count"
            hourly.loc[hourly[count_column].lt(minimum_readings), column] = np.nan

        hourly[TIMESTAMP_COLUMN] = hourly[TIMESTAMP_COLUMN].dt.tz_localize(None)
        hourly[SPLIT_COLUMN] = "live"
        hourly[LIVE_TARGET_COLUMN] = 0
        hourly = hourly.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)

        interpolation_enabled = os.getenv(
            "IOT_HOURLY_INTERPOLATION_ENABLED",
            "false",
        ).strip().lower() in {"1", "true", "yes", "on"}
        maximum_gap_hours = int(os.getenv("IOT_MAX_INTERPOLATED_GAP_HOURS", "8"))

        if interpolation_enabled:
            hourly, interpolation_summaries = interpolate_bounded_hourly_gaps(
                hourly,
                hive_column=HIVE_COLUMN,
                timestamp_column=TIMESTAMP_COLUMN,
                sensor_columns=[
                    "weight_kg",
                    "temperature_c",
                    "humidity_pct",
                    "co2_ppm",
                    "external_temperature_c",
                    "external_humidity_pct",
                    "battery_voltage",
                ],
                required_sensor_columns=list(REQUIRED_SENSOR_COLUMNS),
                max_gap_hours=maximum_gap_hours,
            )
        else:
            hourly["is_imputed_hour"] = False
            hourly["imputed_gap_size_hours"] = 0
            hourly["imputation_method"] = None
            interpolation_summaries = []

        interpolation_lookup = {item["hive_id"]: item for item in interpolation_summaries}

        diagnostics: list[dict[str, Any]] = []
        for hive_id, group in hourly.groupby(HIVE_COLUMN, sort=True):
            group = group.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
            elapsed = group[TIMESTAMP_COLUMN].diff().dt.total_seconds().div(3600.0)
            latest_segment_length = 1
            for value in elapsed.iloc[::-1]:
                if pd.isna(value) or float(value) != 1.0:
                    break
                latest_segment_length += 1
            latest_segment = group.tail(latest_segment_length)
            latest_current_window = latest_segment.tail(168)
            latest_future_window = latest_segment.tail(192)
            latest_current_complete = bool(
                len(latest_current_window) >= 168
                and latest_current_window[list(REQUIRED_SENSOR_COLUMNS)].notna().all(axis=None)
            )
            latest_future_complete = bool(
                len(latest_future_window) >= 192
                and latest_future_window[list(REQUIRED_SENSOR_COLUMNS)].notna().all(axis=None)
            )
            latest_timestamp = group[TIMESTAMP_COLUMN].max()
            latest_window_start = latest_timestamp - pd.Timedelta(hours=191)
            latest_window = group.loc[
                group[TIMESTAMP_COLUMN].between(
                    latest_window_start,
                    latest_timestamp,
                    inclusive="both",
                )
            ]
            latest_window_imputed_hours = int(latest_window["is_imputed_hour"].fillna(False).sum())
            interpolation_summary = interpolation_lookup.get(
                str(hive_id),
                {
                    "interpolation_enabled": interpolation_enabled,
                    "maximum_gap_hours": maximum_gap_hours,
                    "imputed_gap_count": 0,
                    "imputed_hourly_rows": 0,
                    "rejected_gap_count": 0,
                },
            )
            sensor_complete_rows = int(
                group[list(REQUIRED_SENSOR_COLUMNS)].notna().all(axis=1).sum()
            )
            diagnostics.append(
                {
                    "hive_id": str(hive_id),
                    "hourly_rows": len(group),
                    "latest_contiguous_hourly_rows": int(latest_segment_length),
                    "fully_observed_hourly_rows": sensor_complete_rows,
                    "latest_168h_sensor_complete": latest_current_complete,
                    "latest_192h_sensor_complete": latest_future_complete,
                    "latest_hour": latest_timestamp.isoformat(),
                    "minimum_current_hui_hours": 168,
                    "minimum_required_hours": 192,
                    "minimum_readings_per_hour": (self.sensor_settings.minimum_readings_per_hour),
                    "median_readings_per_hour": float(group["readings_in_hour"].median()),
                    "ready_for_current_hui": latest_current_complete,
                    "ready_for_full_hui": latest_future_complete,
                    **interpolation_summary,
                    "latest_192h_imputed_hours": (latest_window_imputed_hours),
                    "latest_192h_imputed_fraction": (
                        latest_window_imputed_hours / max(len(latest_window), 1)
                    ),
                    "imputed_input_active": (latest_window_imputed_hours > 0),
                }
            )

        return hourly, latest_raw, diagnostics

    def _build_live_features(self, hourly: pd.DataFrame) -> pd.DataFrame:
        cfg = self.reviewed_feature_config
        feature_rows, _, _ = build_reviewed_feature_dataset(
            hourly,
            hourly,
            target_column=LIVE_TARGET_COLUMN,
            minimum_history_hours=int(cfg["minimum_history_hours"]),
            weight_windows_hours=[int(value) for value in cfg["weight_windows_hours"]],
            environmental_windows_hours=[
                int(value) for value in cfg["environmental_windows_hours"]
            ],
            weight_delta_hours=[int(value) for value in cfg["weight_delta_hours"]],
            environmental_delta_hours=[int(value) for value in cfg["environmental_delta_hours"]],
            weight_trend_hours=[int(value) for value in cfg["weight_trend_hours"]],
            environmental_trend_hours=[int(value) for value in cfg["environmental_trend_hours"]],
            co2_flatline_std_threshold=float(cfg["co2_flatline_std_threshold"]),
        )
        return feature_rows

    @staticmethod
    def _positive_probability(estimator: Any, features: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise LiveHuiInferenceError(
                "Selected classifier predict_proba did not return two-class probabilities."
            )
        return probabilities[:, -1]

    @staticmethod
    def _apply_calibrator(calibrator: Any, raw_probability: np.ndarray) -> np.ndarray:
        one_dimensional = np.asarray(raw_probability, dtype=float).reshape(-1)
        two_dimensional = one_dimensional.reshape(-1, 1)

        if hasattr(calibrator, "predict_proba"):
            for values in (two_dimensional, one_dimensional):
                try:
                    output = np.asarray(calibrator.predict_proba(values), dtype=float)
                    if output.ndim == 2 and output.shape[1] >= 2:
                        return output[:, -1].clip(0.0, 1.0)
                    if output.size == len(one_dimensional):
                        return output.reshape(-1).clip(0.0, 1.0)
                except (TypeError, ValueError):
                    pass

        if hasattr(calibrator, "predict"):
            for values in (one_dimensional, two_dimensional):
                try:
                    output = np.asarray(calibrator.predict(values), dtype=float).reshape(-1)
                    if len(output) == len(one_dimensional):
                        return output.clip(0.0, 1.0)
                except (TypeError, ValueError):
                    pass

        if hasattr(calibrator, "transform"):
            for values in (one_dimensional, two_dimensional):
                try:
                    output = np.asarray(calibrator.transform(values), dtype=float).reshape(-1)
                    if len(output) == len(one_dimensional):
                        return output.clip(0.0, 1.0)
                except (TypeError, ValueError):
                    pass

        raise LiveHuiInferenceError(
            "The saved probability calibrator does not expose a supported inference method."
        )

    def _score_current_hui(self, feature_rows: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.classifier_features).difference(feature_rows.columns))
        if missing:
            raise LiveHuiInferenceError(
                f"Live feature engineering did not produce classifier features: {missing}"
            )

        result = feature_rows.copy()
        raw_probability = self._positive_probability(
            self.classifier,
            result[self.classifier_features],
        )
        calibrated_probability = self._apply_calibrator(self.calibrator, raw_probability)
        result["raw_probability"] = raw_probability
        result["calibrated_probability"] = calibrated_probability
        result[CURRENT_HUI_COLUMN] = probability_to_hui(
            calibrated_probability,
            probability_anchors=self.probability_anchors,
            hui_anchors=self.hui_anchors,
        )
        result[CURRENT_CLASS_COLUMN] = assign_harvest_readiness_class(
            result[CURRENT_HUI_COLUMN],
            not_ready_upper=self.class_config["not_ready_upper"],
            approaching_upper=self.class_config["approaching_upper"],
            ready_upper=self.class_config["ready_upper"],
        )
        return add_hui_history_features(result)

    def _predict_future_hui(self, current_rows: pd.DataFrame) -> pd.DataFrame:
        result = current_rows.copy()
        for horizon, artifact in self.future_models.items():
            missing = sorted(set(artifact.feature_columns).difference(result.columns))
            if missing:
                raise LiveHuiInferenceError(
                    f"The {horizon}h future-HUI model is missing live features: {missing}"
                )
            ready_mask = result[artifact.feature_columns].notna().all(axis=1)
            result[f"predicted_hui_{horizon}h"] = np.nan
            if ready_mask.any():
                predictions = np.asarray(
                    artifact.estimator.predict(result.loc[ready_mask, artifact.feature_columns]),
                    dtype=float,
                ).clip(0.0, 100.0)
                result.loc[ready_mask, f"predicted_hui_{horizon}h"] = predictions
            result[f"predicted_class_{horizon}h"] = pd.Series(
                pd.NA, index=result.index, dtype="string"
            )
            valid = result[f"predicted_hui_{horizon}h"].notna()
            if valid.any():
                result.loc[valid, f"predicted_class_{horizon}h"] = assign_harvest_readiness_class(
                    result.loc[valid, f"predicted_hui_{horizon}h"],
                    not_ready_upper=self.class_config["not_ready_upper"],
                    approaching_upper=self.class_config["approaching_upper"],
                    ready_upper=self.class_config["ready_upper"],
                ).to_numpy()
        return result

    @staticmethod
    def _latest_aligned_ready_rows(
        predicted_rows: pd.DataFrame,
        diagnostics: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        aligned: list[pd.DataFrame] = []
        updated: list[dict[str, Any]] = []

        for item in diagnostics:
            diagnostic = dict(item)
            hive_id = str(diagnostic["hive_id"])
            latest_hour = pd.Timestamp(diagnostic["latest_hour"])
            hive_rows = predicted_rows.loc[
                predicted_rows[HIVE_COLUMN].astype(str).eq(hive_id)
            ].sort_values(TIMESTAMP_COLUMN)

            latest_available = (
                pd.Timestamp(hive_rows[TIMESTAMP_COLUMN].max()) if not hive_rows.empty else None
            )
            diagnostic["latest_model_ready_hour"] = (
                latest_available.isoformat() if latest_available is not None else None
            )
            diagnostic["latest_model_lag_hours"] = (
                float((latest_hour - latest_available).total_seconds() / 3600.0)
                if latest_available is not None
                else None
            )

            exact = hive_rows.loc[hive_rows[TIMESTAMP_COLUMN].eq(latest_hour)]
            fully_ready = bool(diagnostic.get("ready_for_full_hui"))
            if fully_ready and not exact.empty:
                aligned.append(exact.tail(1))
                diagnostic["live_prediction_ready"] = True
                diagnostic["prediction_timestamp_matches_latest_hour"] = True
            else:
                diagnostic["live_prediction_ready"] = False
                diagnostic["prediction_timestamp_matches_latest_hour"] = False
            updated.append(diagnostic)

        if not aligned:
            return predicted_rows.iloc[0:0].copy(), updated
        return pd.concat(aligned, ignore_index=True), updated

    def _sensor_domain_checks(
        self,
        raw_latest: pd.Series | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        if raw_latest is None or not self.training_sensor_profile:
            return [], False

        checks: list[dict[str, Any]] = []
        for key, limits in self.training_sensor_profile.items():
            value = raw_latest.get(
                {
                    "weight_kg": "weight_kg",
                    "internal_temperature_c": "temperature_c",
                    "co2_ppm": "co2_ppm",
                }[key]
            )
            if pd.isna(value):
                continue
            numeric = float(value)
            outside = numeric < limits["q01"] or numeric > limits["q99"]
            checks.append(
                {
                    "sensor": key,
                    "live_value": numeric,
                    "training_q01": limits["q01"],
                    "training_q99": limits["q99"],
                    "outside_training_range": outside,
                }
            )

        important_shift = any(
            check["sensor"] == "weight_kg" and bool(check["outside_training_range"])
            for check in checks
        )
        return checks, important_shift

    @staticmethod
    def _recent_hui_stability(group: pd.DataFrame, *, window_rows: int = 24) -> float:
        recent = pd.to_numeric(
            group.tail(window_rows)[CURRENT_HUI_COLUMN], errors="coerce"
        ).dropna()
        if len(recent) < 2:
            return 0.0
        standard_deviation = float(recent.std(ddof=0))
        return float(np.clip(100.0 * (1.0 - standard_deviation / 20.0), 0.0, 100.0))

    @staticmethod
    def _recent_hui_slope(group: pd.DataFrame, *, window_rows: int = 6) -> float:
        recent = group.tail(window_rows).copy()
        if len(recent) < 2:
            return 0.0
        elapsed = (
            (recent[TIMESTAMP_COLUMN] - recent[TIMESTAMP_COLUMN].iloc[0])
            .dt.total_seconds()
            .div(3600.0)
        )
        values = pd.to_numeric(recent[CURRENT_HUI_COLUMN], errors="coerce")
        valid = elapsed.notna() & values.notna()
        if valid.sum() < 2 or float(elapsed.loc[valid].max()) == 0.0:
            return 0.0
        slope, _ = np.polyfit(elapsed.loc[valid], values.loc[valid], 1)
        return float(slope)

    @staticmethod
    def _rate_label(slope: float) -> str:
        if slope > 0.5:
            return "Increasing"
        if slope < -0.5:
            return "Decreasing"
        return "Stable"

    @staticmethod
    def _recommended_window(row: pd.Series) -> tuple[str, str]:
        current = float(row[CURRENT_HUI_COLUMN])
        forecasts = {h: float(row[f"predicted_hui_{h}h"]) for h in HORIZONS}
        if current >= 80.0:
            return (
                "Immediate inspection",
                "High current urgency. Inspect the hive promptly and confirm colony and honey conditions before harvesting.",
            )
        if current >= 60.0 or forecasts[24] >= 60.0:
            return (
                "Within 24 hours",
                "Current or 24-hour HUI is in the Ready range. Arrange beekeeper inspection within the next day.",
            )
        if forecasts[48] >= 60.0:
            return (
                "Within 24–48 hours",
                "The 48-hour HUI forecast enters the Ready range. Plan inspection during the next two days.",
            )
        if forecasts[72] >= 60.0:
            return (
                "Within 48–72 hours",
                "The 72-hour HUI forecast enters the Ready range. Continue monitoring and plan inspection within three days.",
            )
        if current >= 40.0 or max(forecasts.values()) >= 40.0:
            return (
                "Continue close monitoring",
                "The hive is approaching harvest readiness, but no forecast reaches the Ready threshold.",
            )
        return (
            "No harvest window indicated",
            "Continue routine monitoring. Current and forecast HUI values remain below the approaching threshold.",
        )

    def _confidence(
        self,
        *,
        hrsi: float,
        completeness: float,
        freshness_minutes: float,
        domain_shift: bool = False,
    ) -> tuple[float, str]:
        calibration_component = (
            100.0
            if bool(self.calibration_gate.get("gate_passed"))
            else 50.0
            if self.calibration_gate.get("selected_method") not in (None, "identity")
            else 25.0
        )
        freshness_factor = (
            1.0 if freshness_minutes <= self.artifact_settings.stale_after_minutes else 0.5
        )
        adjusted_completeness = completeness * freshness_factor
        score = 0.40 * calibration_component + 0.35 * hrsi + 0.25 * adjusted_completeness
        score = float(np.clip(score, 0.0, 100.0))
        if not bool(self.calibration_gate.get("gate_passed")):
            score = min(score, 74.9)
        if domain_shift:
            score = min(score, 49.9)
        label = "Low" if score < 50.0 else "Moderate" if score < 75.0 else "High"
        return score, label

    @staticmethod
    def _contributing_factors(row: pd.Series, *, slope: float) -> list[str]:
        factors: list[str] = []
        delta = row.get("weight_delta_72h_kg")
        if pd.notna(delta):
            delta_value = float(delta)
            if delta_value >= 1.0:
                factors.append("Strong recent 72-hour weight gain")
            elif delta_value > 0.0:
                factors.append("Positive recent hive-weight accumulation")
            elif delta_value <= -1.0:
                factors.append("Recent hive-weight reduction limits readiness")

        relative = row.get("weight_relative_to_max_168h")
        if pd.notna(relative) and float(relative) >= 0.95:
            factors.append("Hive weight remains close to its seven-day maximum")

        variability = row.get("environmental_variability_72h")
        if pd.notna(variability):
            factors.append(
                "Recent environmental conditions contribute to the classifier-derived urgency signal"
            )

        if slope > 0.5:
            factors.append("Current HUI has been increasing over the latest six hourly predictions")
        elif slope < -0.5:
            factors.append("Current HUI has been decreasing over the latest six hourly predictions")
        else:
            factors.append("Current HUI has remained relatively stable recently")
        return factors[:4]

    def _latest_raw_lookup(self, latest_raw: pd.DataFrame) -> dict[str, pd.Series]:
        return {str(row[HIVE_COLUMN]): row for _, row in latest_raw.iterrows()}

    def build_payload(self, raw: pd.DataFrame) -> dict[str, Any]:
        hourly, latest_raw, diagnostics = self.prepare_hourly_history(raw)
        try:
            feature_rows = self._build_live_features(hourly)
        except ValueError as error:
            raise InsufficientLiveHistoryError(
                "No hive currently has enough complete, contiguous hourly sensor history for classifier features.",
                diagnostics=diagnostics,
            ) from error

        current_rows = self._score_current_hui(feature_rows)
        predicted_rows = self._predict_future_hui(current_rows)
        complete_future = (
            predicted_rows[[f"predicted_hui_{horizon}h" for horizon in HORIZONS]]
            .notna()
            .all(axis=1)
        )
        model_ready_rows = predicted_rows.loc[complete_future].copy()
        ready, diagnostics = self._latest_aligned_ready_rows(model_ready_rows, diagnostics)
        if ready.empty:
            raise InsufficientLiveHistoryError(
                "The latest IoT hour does not yet have 192 contiguous, complete hourly observations. "
                "Older model-ready rows are intentionally not returned as live predictions.",
                diagnostics=diagnostics,
            )

        latest_lookup = self._latest_raw_lookup(latest_raw)
        diagnostic_lookup_for_records = {item["hive_id"]: item for item in diagnostics}
        latest_records: list[dict[str, Any]] = []
        for hive_id, group in ready.groupby(HIVE_COLUMN, sort=True):
            group = group.sort_values(TIMESTAMP_COLUMN)
            latest = group.iloc[-1]
            raw_latest = latest_lookup.get(str(hive_id))
            hive_diagnostic = diagnostic_lookup_for_records.get(
                str(hive_id),
                {},
            )
            imputed_input_hours = int(
                hive_diagnostic.get(
                    "latest_192h_imputed_hours",
                    0,
                )
            )
            imputation_applied = imputed_input_hours > 0
            hrsi = self._recent_hui_stability(
                current_rows.loc[current_rows[HIVE_COLUMN].eq(hive_id)]
            )
            slope = self._recent_hui_slope(current_rows.loc[current_rows[HIVE_COLUMN].eq(hive_id)])
            freshness_minutes = float("inf")
            if raw_latest is not None:
                raw_timestamp = pd.Timestamp(raw_latest["source_timestamp_utc"])
                freshness_minutes = max(
                    0.0,
                    (pd.Timestamp.now(tz="UTC") - raw_timestamp).total_seconds() / 60.0,
                )

            completeness_values = []
            if raw_latest is not None:
                for column in (
                    "weight_kg",
                    "temperature_c",
                    "humidity_pct",
                    "co2_ppm",
                ):
                    completeness_values.append(pd.notna(raw_latest.get(column)))
            completeness = (
                100.0 * float(np.mean(completeness_values)) if completeness_values else 0.0
            )
            domain_checks, domain_shift = self._sensor_domain_checks(raw_latest)
            confidence_score, confidence_label = self._confidence(
                hrsi=hrsi,
                completeness=completeness,
                freshness_minutes=freshness_minutes,
                domain_shift=domain_shift,
            )
            window, recommendation = self._recommended_window(latest)
            if domain_shift:
                recommendation = (
                    "Experimental transfer warning: live hive weight is outside the "
                    "historical classifier training range. " + recommendation
                )
            if imputation_applied:
                confidence_score = min(confidence_score, 49.9)
                confidence_label = "Low"
                recommendation = (
                    "Imputed-input research prediction: "
                    f"{imputed_input_hours} missing hourly sensor buckets "
                    "inside the latest 192-hour model window were reconstructed "
                    "through bounded linear interpolation. " + recommendation
                )

            sensor_status = {
                "weight_kg": _json_safe(raw_latest.get("weight_kg"))
                if raw_latest is not None
                else None,
                "internal_temperature_c": _json_safe(raw_latest.get("temperature_c"))
                if raw_latest is not None
                else None,
                "internal_humidity_pct": _json_safe(raw_latest.get("humidity_pct"))
                if raw_latest is not None
                else None,
                "co2_ppm": _json_safe(raw_latest.get("co2_ppm"))
                if raw_latest is not None
                else None,
                "external_temperature_c": _json_safe(raw_latest.get("external_temperature_c"))
                if raw_latest is not None
                else None,
                "external_humidity_pct": _json_safe(raw_latest.get("external_humidity_pct"))
                if raw_latest is not None
                else None,
                "battery_voltage": _json_safe(raw_latest.get("battery_voltage"))
                if raw_latest is not None
                else None,
                "sensor_freshness": (
                    "Fresh"
                    if freshness_minutes <= self.artifact_settings.stale_after_minutes
                    else "Stale"
                ),
                "freshness_minutes": freshness_minutes,
                "input_completeness_percent": completeness,
                "domain_shift_warning": domain_shift,
                "domain_checks": domain_checks,
            }

            factors = self._contributing_factors(latest, slope=slope)
            if imputation_applied:
                factors.insert(
                    0,
                    f"{imputed_input_hours} missing hourly buckets were reconstructed through bounded interpolation",
                )
                factors = factors[:4]
            if domain_shift:
                factors.insert(
                    0,
                    "Live hive weight is outside the historical model training range; interpret HUI as experimental",
                )
                factors = factors[:4]

            latest_records.append(
                {
                    "hive_id": str(hive_id),
                    "prediction_input_mode": (
                        "bounded_hourly_interpolation" if imputation_applied else "observed_only"
                    ),
                    "imputation_applied": imputation_applied,
                    "imputed_hourly_rows": imputed_input_hours,
                    "interpolation_max_gap_hours": int(
                        hive_diagnostic.get(
                            "maximum_gap_hours",
                            int(
                                os.getenv(
                                    "IOT_MAX_INTERPOLATED_GAP_HOURS",
                                    "8",
                                )
                            ),
                        )
                    ),
                    "timestamp": pd.Timestamp(latest[TIMESTAMP_COLUMN]).isoformat(),
                    "source_timestamp_utc": (
                        pd.Timestamp(raw_latest["source_timestamp_utc"]).isoformat()
                        if raw_latest is not None
                        else None
                    ),
                    "raw_probability": float(latest["raw_probability"]),
                    "calibrated_score": float(latest["calibrated_probability"]),
                    "current_hui": float(latest[CURRENT_HUI_COLUMN]),
                    "current_class": str(latest[CURRENT_CLASS_COLUMN]),
                    **{
                        f"predicted_hui_{horizon}h": float(latest[f"predicted_hui_{horizon}h"])
                        for horizon in HORIZONS
                    },
                    **{
                        f"predicted_class_{horizon}h": str(latest[f"predicted_class_{horizon}h"])
                        for horizon in HORIZONS
                    },
                    "hrsi": hrsi,
                    "hrsi_interpretation": (
                        "Stable"
                        if hrsi >= 75.0
                        else "Moderately stable"
                        if hrsi >= 50.0
                        else "Fluctuating"
                    ),
                    "rate_of_change_points_per_hour": slope,
                    "rate_of_change": self._rate_label(slope),
                    "recommended_window": window,
                    "final_recommendation": recommendation,
                    "confidence_score": confidence_score,
                    "prediction_confidence": confidence_label,
                    "domain_shift_warning": domain_shift,
                    "domain_shift_sensors": [
                        check["sensor"]
                        for check in domain_checks
                        if check["outside_training_range"]
                    ],
                    "sensor_status": sensor_status,
                    "contributing_factors": factors,
                }
            )

        series_columns = [
            TIMESTAMP_COLUMN,
            HIVE_COLUMN,
            CURRENT_HUI_COLUMN,
            CURRENT_CLASS_COLUMN,
            "raw_probability",
            "calibrated_probability",
        ]
        diagnostic_lookup = {item["hive_id"]: item for item in diagnostics}
        series_parts: list[pd.DataFrame] = []
        for hive_id in sorted(record["hive_id"] for record in latest_records):
            item = diagnostic_lookup[hive_id]
            latest_hour = pd.Timestamp(item["latest_hour"])
            segment_length = int(item["latest_contiguous_hourly_rows"])
            segment_start = latest_hour - pd.Timedelta(hours=segment_length - 1)
            hive_series = current_rows.loc[
                current_rows[HIVE_COLUMN].astype(str).eq(hive_id)
                & current_rows[TIMESTAMP_COLUMN].between(
                    segment_start, latest_hour, inclusive="both"
                )
            ].sort_values(TIMESTAMP_COLUMN)
            series_parts.append(hive_series.tail(self.artifact_settings.series_rows_per_hive))
        series = (
            pd.concat(series_parts, ignore_index=True)[series_columns].copy()
            if series_parts
            else current_rows.iloc[0:0][series_columns].copy()
        )
        series[TIMESTAMP_COLUMN] = series[TIMESTAMP_COLUMN].map(
            lambda value: pd.Timestamp(value).isoformat()
        )
        series_records = [
            {key: _json_safe(value) for key, value in record.items()}
            for record in series.to_dict(orient="records")
        ]

        available_hives = sorted(record["hive_id"] for record in latest_records)
        domain_shift_detected = any(
            bool(record.get("domain_shift_warning")) for record in latest_records
        )

        return {
            "status": "live_classifier_derived_hui_ready",
            "generated_at": datetime.now(UTC).isoformat(),
            "data_source": "postgresql_iot",
            "data_mode": (
                "historical_replay"
                if self.sensor_settings.history_reference == "database_latest"
                else "live"
            ),
            "history_reference": self.sensor_settings.history_reference,
            "minimum_readings_per_hour": (self.sensor_settings.minimum_readings_per_hour),
            "feature_timezone": self.sensor_settings.feature_timezone,
            "available_hives": available_hives,
            "latest_by_hive": latest_records,
            "hui_series": series_records,
            "hive_diagnostics": diagnostics,
            "models": {
                "current_hui_classifier": "XGBoost 72-hour harvest-risk classifier",
                "future_hui": {
                    str(horizon): {
                        "model": _model_label(artifact.model_name),
                        "feature_count": len(artifact.feature_columns),
                    }
                    for horizon, artifact in self.future_models.items()
                },
            },
            "research_status": {
                "future_hui_gate_passed": bool(self.future_hui_gate.get("gate_passed")),
                "calibration_gate_passed": bool(self.calibration_gate.get("gate_passed")),
                "operational_deployment_allowed": False,
                "domain_shift_detected": domain_shift_detected,
                "hourly_interpolation_enabled": (
                    os.getenv(
                        "IOT_HOURLY_INTERPOLATION_ENABLED",
                        "false",
                    )
                    .strip()
                    .lower()
                    in {"1", "true", "yes", "on"}
                ),
                "imputed_input_prediction": any(
                    bool(record.get("imputation_applied")) for record in latest_records
                ),
                "interpolation_max_gap_hours": int(
                    os.getenv(
                        "IOT_MAX_INTERPOLATED_GAP_HOURS",
                        "8",
                    )
                ),
                "notice": (
                    "Live IoT inference uses the frozen research models. Physical hive inspection "
                    "is required before any harvest decision."
                ),
            },
        }
