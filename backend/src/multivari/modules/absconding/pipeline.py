from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multivari.common.io import read_table, write_parquet
from multivari.common.schema import HIVE_COLUMN, TIMESTAMP_COLUMN
from multivari.common.splitting import join_split_manifest
from multivari.common.targets import make_future_event_target

from .config import AbscondingSettings
from .events import attach_episode_splits, build_event_episodes, evaluate_event_warnings
from .features import (
    absconding_sensor_columns,
    build_absconding_features,
    select_feature_columns,
)
from .metrics import (
    choose_alert_threshold,
    classification_metrics,
    selection_score,
)
from .modeling import (
    build_candidate,
    feature_importance,
    fit_candidate,
    positive_probability,
    stratified_training_sample,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AbscondingPaths:
    backend_root: Path

    @property
    def clean_data(self) -> Path:
        return self.backend_root / "data" / "processed" / "absconding_clean.parquet"

    @property
    def split_manifest(self) -> Path:
        return self.backend_root / "data" / "manifests" / "absconding_split_manifest.parquet"

    @property
    def config(self) -> Path:
        return self.backend_root / "config" / "absconding.yaml"

    @property
    def model_directory(self) -> Path:
        return self.backend_root / "artifacts" / "models" / "absconding"

    @property
    def metrics_directory(self) -> Path:
        return self.backend_root / "artifacts" / "metrics" / "absconding"

    @property
    def report_directory(self) -> Path:
        return self.backend_root / "artifacts" / "reports" / "absconding"

    @property
    def prediction_directory(self) -> Path:
        return self.backend_root / "artifacts" / "predictions" / "absconding"

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.backend_root / path

    def create_output_directories(self) -> None:
        for directory in (
            self.model_directory,
            self.metrics_directory,
            self.report_directory,
            self.prediction_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def run_absconding_pipeline(
    *,
    backend_root: str | Path,
    config_path: str | Path | None = None,
    model_candidates: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    paths = AbscondingPaths(Path(backend_root).resolve())
    settings = AbscondingSettings.from_yaml(config_path or paths.config)
    if model_candidates is not None:
        settings = AbscondingSettings(
            **{
                **settings.__dict__,
                "model_candidates": tuple(model_candidates),
            }
        )
    paths.create_output_directories()
    clean_path = paths.resolve(settings.data_clean_path)
    manifest_path = paths.resolve(settings.data_manifest_path)

    _require_file(
        clean_path,
        "Run: python scripts/run_absconding_data_pipeline.py --input "
        "data/raw/absconding/hive_data_with_features.csv",
    )
    _require_file(
        manifest_path,
        "Run the separate Absconding data pipeline first.",
    )

    clean = read_table(clean_path)
    clean[TIMESTAMP_COLUMN] = pd.to_datetime(clean[TIMESTAMP_COLUMN], errors="raise")
    manifest = read_table(manifest_path)
    manifest[TIMESTAMP_COLUMN] = pd.to_datetime(manifest[TIMESTAMP_COLUMN], errors="raise")

    prepared, feature_names, episodes = prepare_absconding_dataset(clean, manifest, settings)
    split_frames = {
        split: prepared.loc[prepared["split"].eq(split)].copy()
        for split in ("train", "validation", "test")
    }
    _validate_splits(split_frames, settings.target_column)

    X_train = split_frames["train"][feature_names]
    y_train = split_frames["train"][settings.target_column].to_numpy(dtype=int)
    X_validation = split_frames["validation"][feature_names]
    y_validation = split_frames["validation"][settings.target_column].to_numpy(dtype=int)

    comparison: list[dict[str, Any]] = []
    X_comparison, y_comparison = stratified_training_sample(
        X_train,
        y_train,
        maximum_rows=settings.comparison_training_rows,
        random_state=settings.random_state,
    )

    for key in settings.model_candidates:
        try:
            candidate = fit_candidate(build_candidate(key, settings), X_comparison, y_comparison)
            probability = positive_probability(candidate.estimator, X_validation)
            threshold = choose_alert_threshold(
                y_validation,
                probability,
                beta=settings.threshold_beta,
                maximum_alert_fraction=settings.maximum_validation_alert_fraction,
            )
            validation_predictions = _prediction_frame(
                split_frames["validation"], probability, settings
            )
            event_metrics, _ = evaluate_event_warnings(
                validation_predictions,
                episodes,
                threshold=float(threshold["threshold"]),
                horizon_hours=settings.prediction_horizon_hours,
                split="validation",
            )
            metrics = classification_metrics(
                y_validation,
                probability,
                float(threshold["threshold"]),
            )
            score = selection_score(metrics, event_metrics)
            if not threshold["constraint_satisfied"]:
                alert_fraction = max(float(threshold["alert_fraction"]), 1e-12)
                score *= min(1.0, settings.maximum_validation_alert_fraction / alert_fraction)
            comparison.append(
                {
                    "model_key": key,
                    "model_name": candidate.display_name,
                    "model_family": candidate.family,
                    "status": "completed",
                    "selection_score": round(float(score), 8),
                    "threshold_selection": _json_ready(threshold),
                    "validation_metrics": metrics,
                    "validation_event_metrics": event_metrics,
                    "comparison_training_records": len(X_comparison),
                }
            )
        except Exception as error:
            LOGGER.exception(
                "Absconding model candidate '%s' failed; continuing with the remaining candidates.",
                key,
            )
            comparison.append(
                {
                    "model_key": key,
                    "model_name": key.replace("_", " ").title(),
                    "model_family": "Unavailable",
                    "status": "failed",
                    "error": str(error),
                    "selection_score": -1.0,
                    "validation_metrics": {},
                    "validation_event_metrics": {},
                    "comparison_training_records": len(X_comparison),
                }
            )

    successful = [row for row in comparison if row.get("status") == "completed"]
    if not successful:
        raise RuntimeError("Every configured Absconding model candidate failed.")
    eligible = [row for row in successful if row["model_key"] != "dummy_prior"] or successful
    selected_row = max(
        eligible,
        key=lambda row: (
            row["selection_score"],
            row["validation_metrics"].get("pr_auc") or 0,
            row["validation_metrics"].get("f2") or 0,
        ),
    )
    selected_key = selected_row["model_key"]

    # Refit the chosen model on a larger event-preserving sample. The hourly
    # records are highly redundant, so keeping every positive row and a capped,
    # reproducible negative sample gives practical runtime without discarding an
    # absconding warning window.
    X_final_train, y_final_train = stratified_training_sample(
        X_train,
        y_train,
        maximum_rows=settings.final_training_rows,
        random_state=settings.random_state + 1,
    )
    selected = fit_candidate(build_candidate(selected_key, settings), X_final_train, y_final_train)
    selected_validation_probability = positive_probability(selected.estimator, X_validation)
    selected_threshold = choose_alert_threshold(
        y_validation,
        selected_validation_probability,
        beta=settings.threshold_beta,
        maximum_alert_fraction=settings.maximum_validation_alert_fraction,
    )
    alert_threshold = float(selected_threshold["threshold"])
    medium_threshold = min(
        alert_threshold,
        max(0.0, alert_threshold * settings.medium_threshold_ratio),
    )
    selected_validation_predictions = _prediction_frame(
        split_frames["validation"], selected_validation_probability, settings
    )
    selected_event_metrics, _ = evaluate_event_warnings(
        selected_validation_predictions,
        episodes,
        threshold=alert_threshold,
        horizon_hours=settings.prediction_horizon_hours,
        split="validation",
    )
    selected_row = {
        **selected_row,
        "threshold_selection": _json_ready(selected_threshold),
        "validation_metrics": classification_metrics(
            y_validation, selected_validation_probability, alert_threshold
        ),
        "validation_event_metrics": selected_event_metrics,
        "available_training_records": len(X_train),
        "final_training_records": len(X_final_train),
    }

    X_test = split_frames["test"][feature_names]
    y_test = split_frames["test"][settings.target_column].to_numpy(dtype=int)
    test_probability = positive_probability(selected.estimator, X_test)
    test_predictions = _prediction_frame(split_frames["test"], test_probability, settings)
    test_metrics = classification_metrics(y_test, test_probability, alert_threshold)
    test_event_metrics, test_event_details = evaluate_event_warnings(
        test_predictions,
        episodes,
        threshold=alert_threshold,
        horizon_hours=settings.prediction_horizon_hours,
        split="test",
    )

    importances = feature_importance(
        selected.estimator,
        X_validation,
        y_validation,
        feature_names,
        maximum_rows=settings.permutation_importance_rows,
        random_state=settings.random_state,
    )

    all_predictions = _score_all_rows(
        prepared,
        selected.estimator,
        feature_names,
        settings,
        alert_threshold=alert_threshold,
        medium_threshold=medium_threshold,
    )
    latest_risk, hive_details, alerts = _build_hive_outputs(
        all_predictions,
        settings,
        alert_threshold=alert_threshold,
        medium_threshold=medium_threshold,
    )

    exploratory = _build_exploratory_payload(prepared, episodes, settings)
    model_bundle = {
        "module": "absconding",
        "model_key": selected_key,
        "model_name": selected.display_name,
        "model_family": selected.family,
        "estimator": selected.estimator,
        "feature_names": feature_names,
        "settings": settings.to_dict(),
        "alert_threshold": alert_threshold,
        "medium_threshold": medium_threshold,
        "training_frequency": "1 hour",
        "prediction_horizon_hours": settings.prediction_horizon_hours,
        "minimum_history_hours": settings.minimum_history_hours,
        "data_contract": {
            "timestamp": TIMESTAMP_COLUMN,
            "hive": HIVE_COLUMN,
            "sensors": list(absconding_sensor_columns(clean)),
            "training_dataset": settings.data_clean_path,
        },
    }
    joblib.dump(model_bundle, paths.model_directory / "absconding_model_bundle.joblib")

    comparison = _merge_existing_lstm_result(comparison, paths, settings)
    legacy_comparison = _legacy_model_comparison(comparison)
    active_metrics = {
        **test_metrics,
        "model_name": selected.display_name,
        "model_key": selected_key,
        "model_family": selected.family,
        "training_records": len(X_final_train),
        "testing_records": len(X_test),
        "target_column": settings.target_column,
        "threshold": round(alert_threshold, 8),
    }
    completed_legacy = [row for row in legacy_comparison if not row.get("error")]
    best_defence = max(
        completed_legacy, key=lambda row: row.get("defence_score", -1.0), default=None
    )

    dashboard = {
        "summary": {
            "module_name": "Absconding Early Warning",
            "status": "research_model_separate_absconding_dataset",
            "selected_model_key": selected_key,
            "selected_model_name": selected.display_name,
            "selected_model_family": selected.family,
            "active_backend_model": selected.display_name,
            "prediction_horizon_hours": settings.prediction_horizon_hours,
            "minimum_history_hours": settings.minimum_history_hours,
            "source_active_event_rows": int(
                clean.get(settings.active_event_column, pd.Series(dtype="int8")).sum()
            ),
            "source_event_markers": int(clean[settings.event_column].sum()),
            "distinct_event_episodes": len(episodes),
            "future_positive_rows": int(prepared[settings.target_column].sum()),
            "future_positive_rate": round(float(prepared[settings.target_column].mean()), 8),
            "total_records_used": len(prepared),
            "total_hives": int(prepared[HIVE_COLUMN].nunique()),
            "analysis_start": prepared[TIMESTAMP_COLUMN].min().isoformat(),
            "analysis_end": prepared[TIMESTAMP_COLUMN].max().isoformat(),
            "alert_threshold": round(alert_threshold, 8),
            "medium_threshold": round(medium_threshold, 8),
            "high_risk_hives": sum(item["risk_level"] == "High" for item in latest_risk),
            "medium_risk_hives": sum(item["risk_level"] == "Medium" for item in latest_risk),
            "low_risk_hives": sum(item["risk_level"] == "Low" for item in latest_risk),
            "methodology_note": (
                "The Absconding module is trained from its separate labelled historical dataset, "
                f"using event onsets and a {settings.prediction_horizon_hours}-hour future warning "
                "target. The shared common dataset used by the other modules is unchanged. "
                "Results are reported with both row-level and event-level metrics and require "
                "local Sri Lankan biological validation before production use."
            ),
            "training_dataset": settings.data_clean_path,
        },
        "exploratory_analysis": exploratory,
        "model_training": {
            "selected_model": selected_row,
            "model_comparison": sorted(
                comparison, key=lambda row: row["selection_score"], reverse=True
            ),
            "test_metrics": test_metrics,
            "test_event_metrics": test_event_metrics,
            "test_event_details": test_event_details,
            "feature_importance": importances[:25],
            "features_used": feature_names,
            "selection_rule": (
                "Model selection uses validation PR-AUC, F2, precision, and event recall. "
                "The alert threshold is chosen only on validation data and then frozen for test evaluation."
            ),
        },
        # Compatibility fields preserve the three report/dashboard interfaces from
        # the previous version while the canonical V2 data contract remains above.
        "model_metrics": active_metrics,
        "model_comparison": legacy_comparison,
        "model_selection_rationale": {
            "best_model_by_defence_score": (
                best_defence.get("model_name") if best_defence else selected.display_name
            ),
            "lstm_metrics_available": any(
                row.get("model_key") == "lstm_sequence" and row.get("status") == "completed"
                for row in comparison
            ),
            "is_lstm_best": bool(best_defence and best_defence.get("model_key") == "lstm_sequence"),
            "why_lstm_is_defensible": [
                "Absconding develops through ordered changes rather than one isolated reading.",
                "The report used 72-observation windows with stride three for the LSTM experiment.",
                "The LSTM comparison uses real ordered windows and is installed through the optional lstm dependency group.",
                "The live endpoint remains on the validation-selected tabular model unless LSTM is explicitly promoted after defensible evaluation.",
            ],
        },
        "feature_importance": importances[:25],
        "per_hive_absconding_risk": latest_risk,
        "risk_thresholds": {
            "low": f"< {medium_threshold:.6f}",
            "medium": f"{medium_threshold:.6f}–{alert_threshold:.6f}",
            "high": f">= {alert_threshold:.6f}",
            "arm_escalation": f">= {settings.arm_escalation_threshold:.4f} over {settings.arm_change_hours}h",
        },
        "risk_configuration": {
            "low": f"probability < {medium_threshold:.6f}",
            "medium": f"{medium_threshold:.6f} <= probability < {alert_threshold:.6f}",
            "high": f"probability >= {alert_threshold:.6f}",
            "arm_escalation_threshold": settings.arm_escalation_threshold,
        },
        "latest_hive_risk": latest_risk,
        "hive_options": [item["hive_id"] for item in latest_risk],
        "hive_details": hive_details,
        "alerts": alerts,
        "plots": {
            "class_balance": "/api/absconding/images/absconding_class_balance.png",
            "model_comparison": "/api/absconding/images/absconding_model_comparison.png",
            "confusion_matrix": "/api/absconding/images/absconding_confusion_matrix.png",
            "feature_importance": "/api/absconding/images/absconding_feature_importance.png",
        },
        "live_inference": {
            "endpoint": "/api/absconding/iot/live",
            "manual_endpoint": "/api/absconding/predict",
            "method": "GET",
            "required_history_hours": settings.minimum_history_hours,
            "accepted_sensor_columns": list(absconding_sensor_columns(clean)),
            "note": (
                "The backend polls Supabase/PostgreSQL every 10 minutes. Higher-frequency readings "
                "are aggregated to hourly means before the exact training feature pipeline is applied."
            ),
        },
    }

    _save_outputs(
        paths,
        dashboard,
        comparison,
        importances,
        all_predictions,
        latest_risk,
        test_event_details,
    )
    _save_plots(
        paths,
        prepared,
        comparison,
        test_metrics,
        importances,
        settings,
    )
    return dashboard


def prepare_absconding_dataset(
    clean: pd.DataFrame,
    manifest: pd.DataFrame,
    settings: AbscondingSettings,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    target_frame = make_future_event_target(
        clean,
        event_column=settings.event_column,
        horizon_hours=settings.prediction_horizon_hours,
        output_column=settings.target_column,
    )
    featured = build_absconding_features(target_frame, settings)
    prepared = join_split_manifest(featured, manifest)
    prepared = prepared.loc[
        ~prepared["is_boundary_gap"]
        & prepared[settings.target_column].notna()
        & prepared["has_full_absconding_history"].eq(1)
    ].copy()
    prepared[settings.target_column] = prepared[settings.target_column].astype("int8")
    feature_names = select_feature_columns(
        prepared,
        extra_excluded={settings.target_column, settings.event_column},
    )
    episodes = attach_episode_splits(
        build_event_episodes(
            clean,
            event_column=settings.event_column,
            merge_gap_hours=settings.event_merge_gap_hours,
        ),
        join_split_manifest(clean, manifest),
    )
    return prepared, feature_names, episodes


def _score_all_rows(
    prepared: pd.DataFrame,
    estimator: Any,
    feature_names: list[str],
    settings: AbscondingSettings,
    *,
    alert_threshold: float,
    medium_threshold: float,
    batch_size: int = 50_000,
) -> pd.DataFrame:
    # Score in batches and retain only dashboard/inference fields. Copying the
    # complete feature matrix here can require several gigabytes for long hive
    # histories and is unnecessary after probabilities have been calculated.
    probability_parts: list[np.ndarray] = []
    for start in range(0, len(prepared), batch_size):
        stop = min(start + batch_size, len(prepared))
        probability_parts.append(
            positive_probability(estimator, prepared.iloc[start:stop][feature_names])
        )
    probability = (
        np.concatenate(probability_parts) if probability_parts else np.array([], dtype=float)
    )

    explanation_columns = [
        "weight_kg_change_1h",
        "weight_kg_change_6h",
        "weight_kg_change_24h",
        "weight_kg_change_72h",
        "co2_ppm_change_6h",
        "co2_ppm_change_24h",
        "co2_ppm_change_72h",
        "temperature_c_change_6h",
        "temperature_c_change_24h",
        "humidity_pct_change_6h",
        "humidity_pct_change_24h",
        "co2_ppm_z_72h",
        "temperature_c_z_72h",
        "humidity_pct_z_72h",
        "multisensor_instability_index",
        "temperature_deviation_from_35",
        "humidity_deviation_from_optimal",
        "environmental_stress_score",
        "stress_trend_24h",
        "co2_high_flag",
        "rapid_weight_loss_flag",
        "sustained_weight_loss_24h",
        "sustained_weight_loss_72h",
    ]
    keep_columns = [
        HIVE_COLUMN,
        TIMESTAMP_COLUMN,
        "split",
        settings.target_column,
        *absconding_sensor_columns(prepared),
        *[column for column in explanation_columns if column in prepared.columns],
    ]
    keep_columns = list(dict.fromkeys(keep_columns))
    result = prepared[keep_columns].copy()
    result["absconding_probability"] = probability
    result["risk_percentage"] = result["absconding_probability"] * 100
    grouped = result.groupby(HIVE_COLUMN, sort=False)["absconding_probability"]
    result["arm"] = grouped.diff(settings.arm_change_hours).fillna(0.0)
    result["arm_per_hour"] = result["arm"] / max(settings.arm_change_hours, 1)
    result["risk_level"] = np.select(
        [
            result["absconding_probability"].ge(alert_threshold),
            result["absconding_probability"].ge(medium_threshold)
            | result["arm"].ge(settings.arm_escalation_threshold),
        ],
        ["High", "Medium"],
        default="Low",
    )
    return result


def _prediction_frame(
    frame: pd.DataFrame,
    probability: np.ndarray,
    settings: AbscondingSettings,
) -> pd.DataFrame:
    result = frame[[HIVE_COLUMN, TIMESTAMP_COLUMN, "split", settings.target_column]].copy()
    result["absconding_probability"] = probability
    return result


def _build_hive_outputs(
    predictions: pd.DataFrame,
    settings: AbscondingSettings,
    *,
    alert_threshold: float,
    medium_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    latest_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    alerts: list[dict[str, Any]] = []

    for hive_id, group in predictions.groupby(HIVE_COLUMN, sort=True):
        ordered = group.sort_values(TIMESTAMP_COLUMN)
        latest = ordered.iloc[-1]
        factors = _signal_explanations(latest)
        sensor_payload = _sensor_payload(latest)
        latest_item = {
            "hive_id": str(hive_id),
            "hive": str(hive_id),
            "timestamp": latest[TIMESTAMP_COLUMN].isoformat(),
            "probability": round(float(latest["absconding_probability"]), 6),
            "risk_percentage": round(float(latest["risk_percentage"]), 3),
            "risk_level": str(latest["risk_level"]),
            "arm": round(float(latest["arm"]), 6),
            "arm_per_hour": round(float(latest.get("arm_per_hour", 0.0)), 8),
            "arm_trend": _arm_label(float(latest["arm"])),
            "temperature_c": sensor_payload["temperature_c"],
            "humidity_pct": sensor_payload["humidity_pct"],
            "co2_ppm": sensor_payload["co2_ppm"],
            "weight_kg": sensor_payload["weight_kg"],
            "latest_sensor_readings": sensor_payload,
            "signal_explanations": factors,
            "key_factors": factors,
        }
        latest_rows.append(latest_item)
        if latest_item["risk_level"] == "High":
            alerts.append(
                {
                    "hive_id": str(hive_id),
                    "timestamp": latest_item["timestamp"],
                    "risk_level": "High",
                    "probability": latest_item["probability"],
                    "message": "Absconding early-warning threshold reached.",
                }
            )

        timeline = []
        for _, row in ordered.tail(settings.timeline_points_per_hive).iterrows():
            sensors = _sensor_payload(row)
            timeline.append(
                {
                    "timestamp": row[TIMESTAMP_COLUMN].isoformat(),
                    "probability": round(float(row["absconding_probability"]), 6),
                    "risk_probability": round(float(row["absconding_probability"]), 6),
                    "risk_percentage": round(float(row["risk_percentage"]), 3),
                    "arm": round(float(row["arm"]), 6),
                    "arm_per_hour": round(float(row.get("arm_per_hour", 0.0)), 8),
                    "risk_level": str(row["risk_level"]),
                    "environmental_stress_score": sensors["environmental_stress_score"],
                    "temperature_c": sensors["temperature_c"],
                    "humidity_pct": sensors["humidity_pct"],
                    "co2_ppm": sensors["co2_ppm"],
                    "weight_kg": sensors["weight_kg"],
                    "actual_future_label": int(row[settings.target_column]),
                    "actual_next_24h_label": int(row[settings.target_column]),
                    "actual_next_72h_label": None,
                }
            )
        details[str(hive_id)] = {
            "latest": latest_item,
            "timeline": timeline,
            "thresholds": {
                "medium": round(medium_threshold, 8),
                "high": round(alert_threshold, 8),
            },
        }

    order = {"High": 0, "Medium": 1, "Low": 2}
    latest_rows.sort(
        key=lambda item: (order[item["risk_level"]], -item["probability"], item["hive_id"])
    )
    return latest_rows, details, alerts


def _sensor_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "temperature_c": _round_or_none(row.get("temperature_c")),
        "humidity_pct": _round_or_none(row.get("humidity_pct")),
        "co2_ppm": _round_or_none(row.get("co2_ppm")),
        "weight_kg": _round_or_none(row.get("weight_kg")),
        "external_temperature_c": _round_or_none(row.get("external_temperature_c")),
        "external_humidity_pct": _round_or_none(row.get("external_humidity_pct")),
        "internal_external_temperature_difference": _round_or_none(
            row.get("internal_external_temperature_difference")
        ),
        "internal_external_humidity_difference": _round_or_none(
            row.get("internal_external_humidity_difference")
        ),
        "environmental_stress_score": _round_or_none(
            row.get("environmental_stress_score"), digits=6
        ),
        "stress_trend_24h": _round_or_none(row.get("stress_trend_24h"), digits=6),
        "weight_change_1h": _round_or_none(row.get("weight_kg_change_1h")),
        "weight_change_6h": _round_or_none(row.get("weight_kg_change_6h")),
        "weight_change_24h": _round_or_none(row.get("weight_kg_change_24h")),
        "weight_change_72h": _round_or_none(row.get("weight_kg_change_72h")),
        "co2_change_6h": _round_or_none(row.get("co2_ppm_change_6h")),
        "co2_change_24h": _round_or_none(row.get("co2_ppm_change_24h")),
        "co2_change_72h": _round_or_none(row.get("co2_ppm_change_72h")),
        "temp_change_6h": _round_or_none(row.get("temperature_c_change_6h")),
        "temp_change_24h": _round_or_none(row.get("temperature_c_change_24h")),
        "humidity_change_6h": _round_or_none(row.get("humidity_pct_change_6h")),
        "humidity_change_24h": _round_or_none(row.get("humidity_pct_change_24h")),
        "temp_deviation_from_35": _round_or_none(row.get("temperature_deviation_from_35")),
        "humidity_deviation_from_optimal": _round_or_none(
            row.get("humidity_deviation_from_optimal")
        ),
        "co2_high_flag": int(_safe_number(row.get("co2_high_flag"))),
        "rapid_weight_loss_flag": int(_safe_number(row.get("rapid_weight_loss_flag"))),
        "sustained_weight_loss_24h": int(_safe_number(row.get("sustained_weight_loss_24h"))),
        "sustained_weight_loss_72h": int(_safe_number(row.get("sustained_weight_loss_72h"))),
    }


def _signal_explanations(row: pd.Series) -> list[dict[str, Any]]:
    candidates = [
        (
            "Weight trend",
            -_safe_number(row.get("weight_kg_change_72h")),
            f"72-hour weight change: {_round_or_none(row.get('weight_kg_change_72h'))} kg",
        ),
        (
            "CO₂ instability",
            abs(_safe_number(row.get("co2_ppm_z_72h"))),
            f"CO₂ deviation from trailing baseline: {_round_or_none(row.get('co2_ppm_z_72h'))} SD",
        ),
        (
            "Temperature instability",
            abs(_safe_number(row.get("temperature_c_z_72h"))),
            f"Temperature deviation from trailing baseline: {_round_or_none(row.get('temperature_c_z_72h'))} SD",
        ),
        (
            "Humidity instability",
            abs(_safe_number(row.get("humidity_pct_z_72h"))),
            f"Humidity deviation from trailing baseline: {_round_or_none(row.get('humidity_pct_z_72h'))} SD",
        ),
        (
            "Multisensor instability",
            _safe_number(row.get("multisensor_instability_index")),
            f"Combined trailing instability index: {_round_or_none(row.get('multisensor_instability_index'))}",
        ),
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [
        {"factor": factor, "signal_strength": round(max(score, 0.0), 4), "detail": detail}
        for factor, score, detail in candidates[:3]
    ]


def _safe_number(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _build_exploratory_payload(
    prepared: pd.DataFrame,
    episodes: pd.DataFrame,
    settings: AbscondingSettings,
) -> dict[str, Any]:
    split_summary = []
    for split, group in prepared.groupby("split", sort=False):
        split_summary.append(
            {
                "split": str(split),
                "records": len(group),
                "positive_rows": int(group[settings.target_column].sum()),
                "positive_rate": round(float(group[settings.target_column].mean()), 8),
                "event_episodes": int(episodes["split"].eq(split).sum())
                if not episodes.empty
                else 0,
            }
        )

    sensor_effects = []
    for sensor in absconding_sensor_columns(prepared):
        normal = prepared.loc[prepared[settings.target_column].eq(0), sensor].dropna()
        warning = prepared.loc[prepared[settings.target_column].eq(1), sensor].dropna()
        pooled_std = float(normal.std()) if len(normal) > 1 else 0.0
        effect = (float(warning.mean()) - float(normal.mean())) / pooled_std if pooled_std else 0.0
        sensor_effects.append(
            {
                "sensor": sensor,
                "normal_mean": round(float(normal.mean()), 5),
                "pre_event_mean": round(float(warning.mean()), 5),
                "standardized_difference": round(effect, 5),
            }
        )

    episode_rows = []
    for row in episodes.itertuples(index=False):
        episode_rows.append(
            {
                "episode_id": row.episode_id,
                "hive_id": row.hive_id,
                "event_start": pd.Timestamp(row.event_start).isoformat(),
                "event_end": pd.Timestamp(row.event_end).isoformat(),
                "marker_count": int(row.marker_count),
                "split": row.split,
            }
        )
    return {
        "split_summary": split_summary,
        "event_episodes": episode_rows,
        "sensor_effects": sorted(
            sensor_effects, key=lambda item: abs(item["standardized_difference"]), reverse=True
        ),
        "target_definition": (
            f"1 when at least one absconding marker occurs in the next "
            f"{settings.prediction_horizon_hours} hourly observations; otherwise 0."
        ),
        "leakage_controls": [
            "Chronological train/validation/test assignments are loaded from the module-specific Absconding split manifest.",
            "Rows around split boundaries are removed before model fitting.",
            "All lag, change and rolling features use current or past sensor observations only.",
            "Threshold selection uses validation data; test data is evaluated once with the frozen threshold.",
        ],
    }


def _merge_existing_lstm_result(
    comparison: list[dict[str, Any]],
    paths: AbscondingPaths,
    settings: AbscondingSettings,
) -> list[dict[str, Any]]:
    path = paths.metrics_directory / "lstm_comparison.json"
    if not path.is_file():
        return comparison
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return comparison
    if result.get("status") != "completed":
        return comparison
    if result.get("target_column") != settings.target_column:
        return comparison
    merged = [row for row in comparison if row.get("model_key") != "lstm_sequence"]
    merged.append(result)
    return sorted(merged, key=lambda row: row.get("selection_score", -1.0), reverse=True)


def _legacy_model_comparison(comparison: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in comparison:
        metrics = item.get("validation_metrics", {})
        event_metrics = item.get("validation_event_metrics", {})
        defence_score = selection_score(metrics, event_metrics) if metrics else 0.0
        rows.append(
            {
                "model_key": item.get("model_key"),
                "model_name": item.get("model_name"),
                "model_family": item.get("model_family", "—"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score", metrics.get("f1")),
                "pr_auc": metrics.get("pr_auc"),
                "roc_auc": metrics.get("roc_auc"),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "defence_score": round(float(defence_score), 8),
                "event_recall": event_metrics.get("event_recall"),
                "threshold": item.get("threshold_selection", {}).get("threshold"),
                "status": item.get("status", "completed"),
                "error": item.get("error"),
            }
        )
    return sorted(rows, key=lambda row: row.get("defence_score") or -1.0, reverse=True)


def _save_outputs(
    paths: AbscondingPaths,
    dashboard: dict[str, Any],
    comparison: list[dict[str, Any]],
    importances: list[dict[str, Any]],
    predictions: pd.DataFrame,
    latest_risk: list[dict[str, Any]],
    test_event_details: list[dict[str, Any]],
) -> None:
    (paths.report_directory / "absconding_dashboard.json").write_text(
        json.dumps(_json_ready(dashboard), indent=2), encoding="utf-8"
    )
    (paths.metrics_directory / "model_comparison.json").write_text(
        json.dumps(_json_ready(comparison), indent=2), encoding="utf-8"
    )
    pd.json_normalize(comparison).to_csv(
        paths.metrics_directory / "model_comparison.csv", index=False
    )
    pd.DataFrame(importances).to_csv(
        paths.metrics_directory / "feature_importance.csv", index=False
    )
    pd.DataFrame(test_event_details).to_csv(
        paths.metrics_directory / "test_event_detection.csv", index=False
    )
    pd.DataFrame(latest_risk).drop(columns=["signal_explanations"], errors="ignore").to_csv(
        paths.prediction_directory / "latest_risk_per_hive.csv", index=False
    )
    output_columns = [
        TIMESTAMP_COLUMN,
        HIVE_COLUMN,
        "split",
        "absconding_probability",
        "risk_percentage",
        "arm",
        "risk_level",
    ]
    write_parquet(
        predictions[output_columns],
        paths.prediction_directory / "absconding_risk_timeline.parquet",
    )


def _save_plots(
    paths: AbscondingPaths,
    prepared: pd.DataFrame,
    comparison: list[dict[str, Any]],
    test_metrics: dict[str, Any],
    importances: list[dict[str, Any]],
    settings: AbscondingSettings,
) -> None:
    counts = prepared.groupby(["split", settings.target_column]).size().unstack(fill_value=0)
    ax = counts.plot(kind="bar", figsize=(8, 5), logy=True)
    ax.set_title(f"Absconding {settings.prediction_horizon_hours}-hour target balance by split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Rows (log scale)")
    ax.legend(["Normal", "Future-event window"])
    plt.tight_layout()
    plt.savefig(paths.report_directory / "absconding_class_balance.png", dpi=150)
    plt.close()

    ordered = sorted(comparison, key=lambda row: row["selection_score"], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(
        [row["model_name"] for row in reversed(ordered)],
        [row["selection_score"] for row in reversed(ordered)],
    )
    ax.set_xlabel("Validation selection score")
    ax.set_title("Absconding model comparison")
    plt.tight_layout()
    plt.savefig(paths.report_directory / "absconding_model_comparison.png", dpi=150)
    plt.close()

    cm = test_metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix)
    ax.set_xticks([0, 1], labels=["Pred normal", "Pred warning"])
    ax.set_yticks([0, 1], labels=["Actual normal", "Actual warning"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontweight="bold")
    ax.set_title("Absconding test confusion matrix")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.savefig(paths.report_directory / "absconding_confusion_matrix.png", dpi=150)
    plt.close()

    top = [item for item in importances if item["importance"] > 0][:15]
    if top:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            [item["feature"] for item in reversed(top)],
            [item["importance"] for item in reversed(top)],
        )
        ax.set_xlabel("Importance")
        ax.set_title("Selected-model feature importance")
        plt.tight_layout()
        plt.savefig(paths.report_directory / "absconding_feature_importance.png", dpi=150)
        plt.close()


def _validate_splits(split_frames: dict[str, pd.DataFrame], target_column: str) -> None:
    for split, frame in split_frames.items():
        if frame.empty:
            raise ValueError(f"Absconding {split} split is empty after preparation.")
        classes = set(frame[target_column].unique())
        if classes != {0, 1}:
            raise ValueError(
                f"Absconding {split} split does not contain both classes: {sorted(classes)}. "
                "Reliable supervised evaluation is not possible with this split."
            )


def _require_file(path: Path, guidance: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file was not found: {path}. {guidance}")


def _arm_label(value: float) -> str:
    if value >= 0.04:
        return "Increasing"
    if value <= -0.04:
        return "Improving"
    return "Stable"


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(numeric, digits) if np.isfinite(numeric) else None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (str, bool)) else False:
        return None
    return value
