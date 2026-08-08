from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TIMESTAMP_COLUMN = "timestamp"
HIVE_COLUMN = "hive_id"
SPLIT_COLUMN = "split"
CURRENT_HUI_COLUMN = "classifier_derived_hui"
CURRENT_CLASS_COLUMN = "harvest_readiness_class"
HORIZONS = (24, 48, 72)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _classify_hui(value: float) -> str:
    if value < 40.0:
        return "Not Ready"
    if value < 60.0:
        return "Approaching Harvest"
    if value < 80.0:
        return "Ready"
    return "High-Priority Harvest"


def _load_horizon_predictions(report_root: Path, horizon: int) -> pd.DataFrame:
    path = report_root / f"selected_test_predictions_{horizon}h.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing future-HUI prediction file: {path}")

    frame = pd.read_parquet(path)
    required = {
        TIMESTAMP_COLUMN,
        HIVE_COLUMN,
        SPLIT_COLUMN,
        CURRENT_HUI_COLUMN,
        CURRENT_CLASS_COLUMN,
        "predicted_future_hui",
        "predicted_future_class",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    target_column = f"future_classifier_derived_hui_{horizon}h"
    if target_column not in frame.columns:
        raise ValueError(f"{path.name} is missing target column: {target_column}")

    output = frame[
        [
            TIMESTAMP_COLUMN,
            HIVE_COLUMN,
            SPLIT_COLUMN,
            CURRENT_HUI_COLUMN,
            CURRENT_CLASS_COLUMN,
            target_column,
            "predicted_future_hui",
            "predicted_future_class",
            "actual_future_class",
            "prediction_error_points",
        ]
    ].copy()
    output[TIMESTAMP_COLUMN] = pd.to_datetime(output[TIMESTAMP_COLUMN], errors="raise")
    return output.rename(
        columns={
            target_column: f"actual_hui_{horizon}h",
            "predicted_future_hui": f"predicted_hui_{horizon}h",
            "predicted_future_class": f"predicted_class_{horizon}h",
            "actual_future_class": f"actual_class_{horizon}h",
            "prediction_error_points": f"prediction_error_{horizon}h",
        }
    )


def _merge_horizon_predictions(report_root: Path) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    keys = [
        TIMESTAMP_COLUMN,
        HIVE_COLUMN,
        SPLIT_COLUMN,
        CURRENT_HUI_COLUMN,
        CURRENT_CLASS_COLUMN,
    ]

    for horizon in HORIZONS:
        frame = _load_horizon_predictions(report_root, horizon)
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=keys, how="inner", validate="one_to_one")

    if merged is None or merged.empty:
        raise ValueError("No merged future-HUI prediction rows are available.")

    return merged.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)


def _first_existing(columns: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _attach_sensor_values(
    merged: pd.DataFrame,
    feature_dataset_path: Path,
) -> tuple[pd.DataFrame, dict[str, str | None], dict[str, float]]:
    features = pd.read_parquet(feature_dataset_path)
    features[TIMESTAMP_COLUMN] = pd.to_datetime(features[TIMESTAMP_COLUMN], errors="raise")
    columns = set(features.columns)

    sensor_columns = {
        "weight_kg": _first_existing(columns, ["weight_kg_current", "weight_current"]),
        "internal_temperature_c": _first_existing(
            columns,
            ["temperature_c_current", "internal_temperature_c_current"],
        ),
        "internal_humidity_pct": _first_existing(
            columns,
            [
                "humidity_pct_current",
                "humidity_percent_current",
                "humidity_current",
                "internal_humidity_pct_current",
            ],
        ),
        "co2_ppm": _first_existing(columns, ["co2_ppm_current", "co2_current"]),
        "weight_delta_72h_kg": _first_existing(columns, ["weight_delta_72h_kg"]),
        "weight_relative_to_max_168h": _first_existing(
            columns,
            ["weight_relative_to_max_168h"],
        ),
        "environmental_variability_72h": _first_existing(
            columns,
            ["environmental_variability_72h"],
        ),
    }

    selected = [TIMESTAMP_COLUMN, HIVE_COLUMN, SPLIT_COLUMN]
    selected.extend(
        column
        for column in sensor_columns.values()
        if column is not None and column not in selected
    )

    feature_subset = features[selected].copy()
    output = merged.merge(
        feature_subset,
        on=[TIMESTAMP_COLUMN, HIVE_COLUMN, SPLIT_COLUMN],
        how="left",
        validate="one_to_one",
    )

    thresholds: dict[str, float] = {}
    train = features.loc[features[SPLIT_COLUMN].eq("train")]
    for key in (
        "weight_delta_72h_kg",
        "weight_relative_to_max_168h",
        "environmental_variability_72h",
    ):
        column = sensor_columns[key]
        if column is not None:
            thresholds[f"{key}_median"] = float(
                pd.to_numeric(train[column], errors="coerce").median()
            )

    return output, sensor_columns, thresholds


def _recent_hui_stability(group: pd.DataFrame, *, window_rows: int = 24) -> float:
    recent = pd.to_numeric(
        group.tail(window_rows)[CURRENT_HUI_COLUMN],
        errors="coerce",
    ).dropna()
    if len(recent) < 2:
        return 0.0

    standard_deviation = float(recent.std(ddof=0))
    return float(np.clip(100.0 * (1.0 - standard_deviation / 20.0), 0.0, 100.0))


def _recent_hui_slope(group: pd.DataFrame, *, window_rows: int = 6) -> float:
    recent = group.tail(window_rows).copy()
    if len(recent) < 2:
        return 0.0

    elapsed = (
        (recent[TIMESTAMP_COLUMN] - recent[TIMESTAMP_COLUMN].iloc[0]).dt.total_seconds().div(3600.0)
    )
    values = pd.to_numeric(recent[CURRENT_HUI_COLUMN], errors="coerce")
    valid = elapsed.notna() & values.notna()
    if valid.sum() < 2 or float(elapsed[valid].max()) == 0.0:
        return 0.0

    slope, _ = np.polyfit(elapsed[valid].to_numpy(), values[valid].to_numpy(), 1)
    return float(slope)


def _rate_label(slope: float) -> str:
    if slope > 0.5:
        return "Increasing"
    if slope < -0.5:
        return "Decreasing"
    return "Stable"


def _recommended_window(row: pd.Series) -> tuple[str, str]:
    current = float(row[CURRENT_HUI_COLUMN])
    forecasts = {h: float(row[f"predicted_hui_{h}h"]) for h in HORIZONS}

    if current >= 80.0:
        return (
            "Immediate inspection",
            "High current urgency. Inspect the hive promptly before deciding to harvest.",
        )
    if current >= 60.0:
        return (
            "Within 24 hours",
            "Current HUI is in the Ready range. Conduct a beekeeper inspection within 24 hours.",
        )
    if forecasts[24] >= 60.0:
        return (
            "Within 24 hours",
            "The 24-hour HUI forecast enters the Ready range. Plan inspection within the next day.",
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
    *,
    calibration_gate: dict[str, Any],
    hrsi: float,
    completeness: float,
) -> tuple[float, str]:
    calibration_gate_passed = bool(calibration_gate.get("gate_passed"))

    if calibration_gate_passed:
        calibration_component = 100.0
    elif calibration_gate.get("selected_method") not in (None, "identity"):
        calibration_component = 50.0
    else:
        calibration_component = 25.0

    score = 0.40 * calibration_component + 0.35 * hrsi + 0.25 * completeness
    score = float(np.clip(score, 0.0, 100.0))

    # A High label would overstate the evidence while the probability-
    # calibration gate is limited. Preserve the transparent heuristic
    # score, but cap it within the Moderate band until that gate passes.
    if not calibration_gate_passed:
        score = min(score, 74.9)

    if score < 50.0:
        label = "Low"
    elif score < 75.0:
        label = "Moderate"
    else:
        label = "High"
    return score, label


def _contributing_factors(
    row: pd.Series,
    *,
    sensor_columns: dict[str, str | None],
    thresholds: dict[str, float],
    slope: float,
) -> list[str]:
    factors: list[str] = []

    delta_column = sensor_columns["weight_delta_72h_kg"]
    if delta_column is not None and pd.notna(row.get(delta_column)):
        delta = float(row[delta_column])
        if delta >= 1.0:
            factors.append("Strong recent 72-hour weight gain")
        elif delta > 0.0:
            factors.append("Positive recent hive-weight accumulation")
        elif delta <= -1.0:
            factors.append("Recent hive-weight reduction limits readiness")

    relative_column = sensor_columns["weight_relative_to_max_168h"]
    if relative_column is not None and pd.notna(row.get(relative_column)):
        relative = float(row[relative_column])
        median = thresholds.get("weight_relative_to_max_168h_median", 0.0)
        if relative >= max(0.95, median):
            factors.append("Hive weight remains close to its seven-day maximum")

    variability_column = sensor_columns["environmental_variability_72h"]
    if variability_column is not None and pd.notna(row.get(variability_column)):
        variability = float(row[variability_column])
        median = thresholds.get("environmental_variability_72h_median")
        if median is not None and variability <= median:
            factors.append("Recent environmental conditions are comparatively stable")
        elif median is not None:
            factors.append("Environmental variability reduces prediction confidence")

    if slope > 0.5:
        factors.append("Current HUI has been increasing during the recent HUI-history window")
    elif slope < -0.5:
        factors.append("Current HUI has been decreasing during the recent HUI-history window")
    else:
        factors.append("Current HUI has remained relatively stable recently")

    if not factors:
        factors.append("Prediction is driven by the selected classifier and recent HUI history")

    return factors[:4]


def _build_latest_records(
    data: pd.DataFrame,
    *,
    sensor_columns: dict[str, str | None],
    thresholds: dict[str, float],
    calibration_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for hive_id, group in data.groupby(HIVE_COLUMN, sort=True):
        group = group.sort_values(TIMESTAMP_COLUMN)
        latest = group.iloc[-1]
        hrsi = _recent_hui_stability(group)
        slope = _recent_hui_slope(group)
        rate = _rate_label(slope)
        window, recommendation = _recommended_window(latest)

        available_sensor_values = []
        for key in ("weight_kg", "internal_temperature_c", "internal_humidity_pct", "co2_ppm"):
            column = sensor_columns[key]
            if column is not None:
                available_sensor_values.append(pd.notna(latest.get(column)))
        completeness = (
            100.0 * float(np.mean(available_sensor_values)) if available_sensor_values else 0.0
        )
        confidence_score, confidence_label = _confidence(
            calibration_gate=calibration_gate,
            hrsi=hrsi,
            completeness=completeness,
        )

        sensor_status = {
            "weight_kg": _json_safe(latest.get(sensor_columns["weight_kg"]))
            if sensor_columns["weight_kg"]
            else None,
            "internal_temperature_c": _json_safe(
                latest.get(sensor_columns["internal_temperature_c"])
            )
            if sensor_columns["internal_temperature_c"]
            else None,
            "internal_humidity_pct": _json_safe(latest.get(sensor_columns["internal_humidity_pct"]))
            if sensor_columns["internal_humidity_pct"]
            else None,
            "co2_ppm": _json_safe(latest.get(sensor_columns["co2_ppm"]))
            if sensor_columns["co2_ppm"]
            else None,
            "external_temperature_c": None,
            "external_humidity_pct": None,
            "sensor_freshness": "Historical held-out record",
            "battery_status": "Not available in the research dataset",
            "input_completeness_percent": completeness,
        }

        record: dict[str, Any] = {
            "hive_id": str(hive_id),
            "timestamp": latest[TIMESTAMP_COLUMN].isoformat(),
            "current_hui": float(latest[CURRENT_HUI_COLUMN]),
            "current_class": str(latest[CURRENT_CLASS_COLUMN]),
            "predicted_hui_24h": float(latest["predicted_hui_24h"]),
            "predicted_class_24h": str(latest["predicted_class_24h"]),
            "predicted_hui_48h": float(latest["predicted_hui_48h"]),
            "predicted_class_48h": str(latest["predicted_class_48h"]),
            "predicted_hui_72h": float(latest["predicted_hui_72h"]),
            "predicted_class_72h": str(latest["predicted_class_72h"]),
            "hrsi": hrsi,
            "hrsi_interpretation": (
                "Stable" if hrsi >= 75.0 else "Moderately stable" if hrsi >= 50.0 else "Fluctuating"
            ),
            "rate_of_change_points_per_hour": slope,
            "rate_of_change": rate,
            "recommended_window": window,
            "final_recommendation": recommendation,
            "confidence_score": confidence_score,
            "prediction_confidence": confidence_label,
            "sensor_status": sensor_status,
            "contributing_factors": _contributing_factors(
                latest,
                sensor_columns=sensor_columns,
                thresholds=thresholds,
                slope=slope,
            ),
        }
        records.append(record)

    return records


def _series_records(data: pd.DataFrame, *, rows_per_hive: int) -> list[dict[str, Any]]:
    keep_columns = [
        TIMESTAMP_COLUMN,
        HIVE_COLUMN,
        CURRENT_HUI_COLUMN,
        CURRENT_CLASS_COLUMN,
        "predicted_hui_24h",
        "predicted_hui_48h",
        "predicted_hui_72h",
        "predicted_class_24h",
        "predicted_class_48h",
        "predicted_class_72h",
    ]
    selected = (
        data.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN])
        .groupby(HIVE_COLUMN, group_keys=False, sort=True)
        .tail(rows_per_hive)[keep_columns]
        .copy()
    )
    selected[TIMESTAMP_COLUMN] = selected[TIMESTAMP_COLUMN].map(lambda value: value.isoformat())
    return [
        {key: _json_safe(value) for key, value in record.items()}
        for record in selected.to_dict(orient="records")
    ]


def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    report_root = backend_root / "artifacts/reports/harvesting/reviewed/classifier_derived_hui"
    calibration_root = (
        backend_root / "artifacts/reports/harvesting/reviewed/probability_calibration"
    )
    research_model_root = backend_root / "artifacts/reports/harvesting/reviewed/research_models"

    regression_summary = _read_json(report_root / "future_hui_regression_summary.json")
    regression_gate = _read_json(report_root / "future_hui_regression_gate.json")
    hui_definition = _read_json(report_root / "classifier_derived_hui_definition.json")
    calibration_summary = _read_json(calibration_root / "probability_calibration_summary.json")
    calibration_gate = _read_json(calibration_root / "probability_calibration_gate.json")
    classifier_metrics = _read_json(research_model_root / "selected_model_metrics.json")

    comparison_path = calibration_root / "calibration_method_comparison.csv"
    distribution_path = report_root / "classifier_derived_hui_distribution.csv"
    feature_importance_path = research_model_root / "selected_feature_importance.csv"
    feature_dataset_path = backend_root / "data/processed/harvest_reviewed_feature_dataset.parquet"

    merged = _merge_horizon_predictions(report_root)
    merged, sensor_columns, thresholds = _attach_sensor_values(
        merged,
        feature_dataset_path,
    )

    latest_records = _build_latest_records(
        merged,
        sensor_columns=sensor_columns,
        thresholds=thresholds,
        calibration_gate=calibration_gate,
    )
    series_records = _series_records(merged, rows_per_hive=168)

    calibration_comparison = pd.read_csv(comparison_path)
    calibration_comparison = calibration_comparison.loc[calibration_comparison["status"].eq("ok")]
    calibration_records = [
        {key: _json_safe(value) for key, value in record.items()}
        for record in calibration_comparison.to_dict(orient="records")
    ]

    distribution = pd.read_csv(distribution_path)
    distribution_records = [
        {key: _json_safe(value) for key, value in record.items()}
        for record in distribution.to_dict(orient="records")
    ]

    feature_importance = pd.read_csv(feature_importance_path).head(12)
    feature_records = [
        {key: _json_safe(value) for key, value in record.items()}
        for record in feature_importance.to_dict(orient="records")
    ]

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "classifier_derived_hui_viva_dashboard_ready",
        "research_scope": {
            "ready_for_viva_research_dashboard": bool(
                regression_gate.get("ready_for_viva_research_dashboard")
            ),
            "ready_for_operational_deployment": False,
            "current_hui_definition": (
                "Classifier-derived relative harvest urgency on a 0–100 scale. "
                "It is not a literal probability percentage or direct honey-maturity label."
            ),
            "historical_demo_notice": (
                "The exported hive records are held-out historical test rows. "
                "The same saved models can later be called by the live IoT inference API."
            ),
        },
        "research_gate": regression_gate,
        "research_status": {
            "operational_deployment_allowed": False,
            "probability_calibration_operationally_validated": bool(
                calibration_gate.get("gate_passed")
            ),
            "future_hui_research_gate_passed": bool(regression_gate.get("gate_passed")),
        },
        "hui_definition": hui_definition,
        "classifier_evaluation": classifier_metrics,
        "calibration": {
            "summary": calibration_summary,
            "gate": calibration_gate,
            "comparison": calibration_records,
        },
        "future_hui_regression": {
            "summary": regression_summary,
            "gate": regression_gate,
        },
        "hui_distribution": distribution_records,
        "available_hives": sorted(merged[HIVE_COLUMN].astype(str).unique().tolist()),
        "latest_by_hive": latest_records,
        "historical_test_series": series_records,
        "top_classifier_features": feature_records,
        "decision_support_definition": {
            "hrsi": ("100 × (1 − recent 24-hour HUI standard deviation / 20), clipped to 0–100."),
            "rate_of_change": (
                "Least-squares HUI slope across the latest six hourly records; "
                "Increasing above +0.5 points/hour, Decreasing below −0.5, otherwise Stable."
            ),
            "confidence": (
                "40% calibration evidence, 35% recent HUI stability and 25% input completeness. "
                "This is prototype evidence confidence, not independently validated reliability. "
                "The score and label are capped in the Moderate band while the calibration gate is limited."
            ),
            "classes": {
                "not_ready": [0, 40],
                "approaching_harvest": [40, 60],
                "ready": [60, 80],
                "high_priority_harvest": [80, 100],
            },
        },
    }

    output_path = (
        backend_root
        / "../frontend/public/data/harvesting-research/classifier-derived-hui-viva-dashboard.json"
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, default=_json_safe),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "classifier_derived_hui_viva_dashboard_exported",
                "output_path": str(output_path),
                "hive_count": len(payload["available_hives"]),
                "series_rows": len(series_records),
                "future_hui_gate_passed": bool(regression_gate.get("gate_passed")),
                "operational_deployment_allowed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
