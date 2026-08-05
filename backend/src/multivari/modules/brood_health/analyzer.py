from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .scoring import (
    HEALTH_LEVEL_ORDER,
    BroodHealthScoreConfig,
    classify_health_level,
    compute_score_components,
)

SENSOR_COLUMNS = ("temperature_c", "humidity_pct", "co2_ppm", "weight_kg")
HEALTH_LEVELS = (
    {"level": "Critical", "minimum": 1.0, "maximum": 40.0, "rule": "1 ≤ score < 40"},
    {"level": "Poor", "minimum": 40.0, "maximum": 60.0, "rule": "40 ≤ score < 60"},
    {"level": "Good", "minimum": 60.0, "maximum": 80.0, "rule": "60 ≤ score < 80"},
    {"level": "Excellent", "minimum": 80.0, "maximum": 100.0, "rule": "80 ≤ score ≤ 100"},
)


@dataclass(frozen=True)
class ConditionScoreConfig:
    stability_window_hours: int = 24
    trend_window_hours: int = 12
    minimum_history_for_stability: int = 6


def classify_stability(score: float) -> str:
    value = float(score)
    if value >= 75.0:
        return "High"
    if value >= 50.0:
        return "Moderate"
    return "Low"


def classify_trend(slope: float) -> str:
    value = float(slope)
    if value >= 2.0:
        return "Rapid Improving"
    if value >= 0.5:
        return "Slow Improving"
    if value > -0.5:
        return "Stable"
    if value > -2.0:
        return "Slow Declining"
    return "Rapid Declining"


def compute_condition_components(
    frame: pd.DataFrame,
    *,
    score_config: BroodHealthScoreConfig | None = None,
) -> pd.DataFrame:
    return compute_score_components(frame, config=score_config)


def _rolling_endpoint_slope(values: pd.Series, window: int) -> pd.Series:
    lagged = values.shift(window)
    return (values - lagged) / float(window)


def add_stability_and_trend(
    frame: pd.DataFrame,
    *,
    config: ConditionScoreConfig | None = None,
) -> pd.DataFrame:
    """Add BHSI and Rate of Development using only current and past scores."""

    cfg = config or ConditionScoreConfig()
    out = frame.sort_values(["hive_id", "timestamp"]).reset_index(drop=True).copy()
    grouped = out.groupby("hive_id", sort=False)["brood_health_score"]

    min_periods = max(cfg.minimum_history_for_stability, cfg.stability_window_hours // 4)
    rolling_std = grouped.transform(
        lambda values: values.rolling(cfg.stability_window_hours, min_periods=min_periods).std(ddof=0)
    )
    rolling_min = grouped.transform(
        lambda values: values.rolling(cfg.stability_window_hours, min_periods=min_periods).min()
    )
    rolling_max = grouped.transform(
        lambda values: values.rolling(cfg.stability_window_hours, min_periods=min_periods).max()
    )
    slope = grouped.transform(lambda values: _rolling_endpoint_slope(values, cfg.trend_window_hours))
    slope = slope.fillna(0.0)

    rolling_range = rolling_max - rolling_min
    # BHSI measures environmental stability, not health level. A stable poor hive can
    # therefore have a high BHSI; the dashboard must interpret both measures together.
    bhsi = 100.0 - (1.8 * rolling_std.fillna(12.0) + 0.55 * rolling_range.fillna(20.0) + 4.0 * slope.abs())
    out["bhsi"] = bhsi.clip(0.0, 100.0)
    out["stability_level"] = out["bhsi"].map(classify_stability)
    out["rod_points_per_hour"] = slope
    out["trend_label"] = out["rod_points_per_hour"].map(classify_trend)
    return out


def compute_condition_history(
    frame: pd.DataFrame,
    *,
    score_config: BroodHealthScoreConfig | None = None,
) -> pd.DataFrame:
    required = {"hive_id", "timestamp", *SENSOR_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Condition history is missing columns: {missing}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["hive_id", "timestamp", *SENSOR_COLUMNS])
    out = compute_condition_components(out, score_config=score_config)
    return add_stability_and_trend(out)


def build_warning_payload(
    *,
    forecast_score: float,
    current_condition_score: float,
    bhsi: float,
    rod_points_per_hour: float,
    forecast_drop_points: float | None = None,
    unhealthy_probability: float | None = None,  # Deprecated compatibility input.
    domain_shift_warnings: list[str] | None = None,
    history_sufficient: bool = True,
) -> dict[str, Any]:
    """Combine score, level, BHSI and RoD into an actionable early warning."""

    forecast_level = classify_health_level(forecast_score)
    severity = {"Excellent": 0, "Good": 1, "Poor": 2, "Critical": 3}[forecast_level]
    reasons: list[str] = []

    if forecast_level == "Critical":
        reasons.append("The predicted minimum Brood Health Score within the forecast window is critical.")
    elif forecast_level == "Poor":
        reasons.append("The predicted minimum Brood Health Score within the forecast window is poor.")

    if current_condition_score < 40.0:
        severity = max(severity, 3)
        reasons.append("The current sensor-derived Brood Health Score is critical.")
    elif current_condition_score < 60.0:
        severity = max(severity, 2)
        reasons.append("The current sensor-derived Brood Health Score is poor.")

    drop = float(forecast_drop_points or 0.0)
    if drop >= 20.0:
        severity = max(severity, 3)
        reasons.append(f"The forecast indicates a large score reduction of {drop:.1f} points.")
    elif drop >= 10.0:
        severity = max(severity, 2)
        reasons.append(f"The forecast indicates a meaningful score reduction of {drop:.1f} points.")

    if bhsi < 35.0:
        severity = max(severity, 2)
        reasons.append("BHSI indicates low short-term environmental stability.")
    elif bhsi < 50.0:
        severity = max(severity, 1)
        reasons.append("BHSI indicates moderate environmental instability.")

    if rod_points_per_hour <= -2.0:
        severity = max(severity, 2)
        reasons.append("RoD indicates rapid deterioration in the recent score trajectory.")
    elif rod_points_per_hour <= -0.5:
        severity = max(severity, 1)
        reasons.append("RoD indicates a slowly declining recent score trajectory.")

    domain_shift_warnings = domain_shift_warnings or []
    if domain_shift_warnings:
        severity = max(severity, 1)
        reasons.append("Live sensor values fall outside the central historical training range.")
    if not history_sufficient:
        severity = max(severity, 1)
        reasons.append("The available live history is shorter than the recommended 72 hours.")

    level = {0: "Excellent", 1: "Good", 2: "Poor", 3: "Critical"}[severity]
    actions = {
        "Excellent": ["Continue routine monitoring and verify sensor freshness."],
        "Good": ["Review the next reading and inspect any sensor trend moving away from its usual range."],
        "Poor": ["Inspect the colony soon and verify ventilation, humidity, CO₂, weight trend and sensor calibration."],
        "Critical": ["Perform an immediate physical hive inspection and confirm the alert with direct brood observations."],
    }[level]
    if not reasons:
        reasons.append("Current score, forecast score, BHSI and RoD remain within the configured operating range.")

    return {
        "level": level,
        "title": f"{level} brood-health warning",
        "summary": (
            f"Current score {current_condition_score:.1f}/100; predicted minimum score "
            f"{forecast_score:.1f}/100 within the forecast window."
        ),
        "reasons": reasons,
        "recommended_actions": actions,
        "requires_physical_confirmation": True,
    }


__all__ = [
    "HEALTH_LEVELS",
    "HEALTH_LEVEL_ORDER",
    "ConditionScoreConfig",
    "add_stability_and_trend",
    "build_warning_payload",
    "classify_health_level",
    "classify_stability",
    "classify_trend",
    "compute_condition_components",
    "compute_condition_history",
]
