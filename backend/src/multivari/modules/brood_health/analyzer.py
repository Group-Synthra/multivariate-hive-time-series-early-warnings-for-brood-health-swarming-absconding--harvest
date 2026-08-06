from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .scoring import (
    HEALTH_LEVEL_ORDER,
    HEALTH_LEVEL_RULES,
    BroodHealthScoreConfig,
    classify_health_level,
    compute_score_components,
)

SENSOR_COLUMNS = ("temperature_c", "humidity_pct", "co2_ppm", "weight_kg")
HEALTH_LEVELS = HEALTH_LEVEL_RULES


@dataclass(frozen=True)
class ConditionScoreConfig:
    stability_window_hours: int = 6
    trend_window_hours: int = 4
    minimum_history_for_stability: int = 3
    temperature_variability_tolerance: float = 1.5
    humidity_variability_tolerance: float = 8.0
    log_co2_variability_tolerance: float = 0.30


def classify_stability(score: float) -> str:
    value = float(score)
    if value >= 70.0:
        return "High"
    if value >= 40.0:
        return "Moderate"
    return "Low"


def classify_trend(slope: float) -> str:
    value = float(slope)
    if value > 3.0:
        return "Rapid Improving"
    if value > 0.5:
        return "Slow Improving"
    if value >= -0.5:
        return "Stable"
    if value >= -3.0:
        return "Slow Declining"
    return "Rapid Declining"


def _rolling_linear_slope(values: pd.Series, window: int) -> pd.Series:
    def slope(array: np.ndarray) -> float:
        valid = np.asarray(array, dtype=float)
        valid = valid[np.isfinite(valid)]
        if valid.size < 2:
            return 0.0
        x = np.arange(valid.size, dtype=float)
        return float(np.polyfit(x, valid, 1)[0])

    return values.rolling(window, min_periods=max(2, window // 2)).apply(
        slope, raw=True
    )


def add_stability_and_trend(
    frame: pd.DataFrame,
    *,
    config: ConditionScoreConfig | None = None,
) -> pd.DataFrame:
    """Add BHSI and Rate of Deterioration using current and past observations only.

    BHSI directly measures six-hour variability in internal temperature, humidity and
    log-transformed CO2. RoD is the rolling four-hour linear slope of the current score.
    """

    cfg = config or ConditionScoreConfig()
    out = frame.sort_values(["hive_id", "timestamp"]).reset_index(drop=True).copy()

    temp_std = out.groupby("hive_id", sort=False)["temperature_c"].transform(
        lambda values: values.rolling(
            cfg.stability_window_hours,
            min_periods=cfg.minimum_history_for_stability,
        ).std(ddof=0)
    )
    humidity_std = out.groupby("hive_id", sort=False)["humidity_pct"].transform(
        lambda values: values.rolling(
            cfg.stability_window_hours,
            min_periods=cfg.minimum_history_for_stability,
        ).std(ddof=0)
    )
    log_co2 = np.log1p(pd.to_numeric(out["co2_ppm"], errors="coerce").clip(lower=0.0))
    log_co2_std = log_co2.groupby(out["hive_id"], sort=False).transform(
        lambda values: values.rolling(
            cfg.stability_window_hours,
            min_periods=cfg.minimum_history_for_stability,
        ).std(ddof=0)
    )

    normalized_variability = (
        temp_std / cfg.temperature_variability_tolerance
        + humidity_std / cfg.humidity_variability_tolerance
        + log_co2_std / cfg.log_co2_variability_tolerance
    ) / 3.0
    bhsi = 100.0 * np.exp(-normalized_variability.clip(lower=0.0))
    out["bhsi"] = bhsi.fillna(50.0).clip(0.0, 100.0)
    out["stability_level"] = out["bhsi"].map(classify_stability)

    slope = out.groupby("hive_id", sort=False)["brood_health_score"].transform(
        lambda values: _rolling_linear_slope(values, cfg.trend_window_hours)
    )
    out["rod_points_per_hour"] = slope.fillna(0.0)
    out["trend_label"] = out["rod_points_per_hour"].map(classify_trend)
    return out


def compute_condition_history(
    frame: pd.DataFrame,
    *,
    score_config: BroodHealthScoreConfig | None = None,
    condition_config: ConditionScoreConfig | None = None,
) -> pd.DataFrame:
    required = {"hive_id", "timestamp", *SENSOR_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Condition history is missing columns: {missing}")

    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["hive_id", "timestamp", *SENSOR_COLUMNS])
    out = compute_score_components(out, config=score_config)
    return add_stability_and_trend(out, config=condition_config)


def build_warning_payload(
    *,
    exact_forecast_score: float,
    safety_minimum_score: float,
    current_condition_score: float,
    bhsi: float,
    rod_points_per_hour: float,
    exact_forecast_drop_points: float | None = None,
    safety_drop_points: float | None = None,
    domain_shift_warnings: list[str] | None = None,
    history_sufficient: bool = True,
) -> dict[str, Any]:
    """Combine exact +6 h forecast, safety minimum, BHSI and RoD."""

    exact_level = classify_health_level(exact_forecast_score)
    minimum_level = classify_health_level(safety_minimum_score)
    severity_lookup = {"Excellent": 0, "Good": 1, "Poor": 2, "Critical": 3}
    severity = max(severity_lookup[exact_level], severity_lookup[minimum_level])
    reasons: list[str] = []

    if exact_level in {"Poor", "Critical"}:
        reasons.append(
            f"The predicted score exactly at the forecast horizon is {exact_level.lower()}."
        )
    if minimum_level in {"Poor", "Critical"} and minimum_level != exact_level:
        reasons.append(
            f"The predicted trajectory reaches a {minimum_level.lower()} safety minimum "
            "before the forecast horizon."
        )

    current_level = classify_health_level(current_condition_score)
    if current_level in {"Poor", "Critical"}:
        severity = max(severity, severity_lookup[current_level])
        reasons.append(
            f"The current sensor-derived Brood Health Score is {current_level.lower()}."
        )

    exact_drop = max(0.0, float(exact_forecast_drop_points or 0.0))
    safety_drop = max(0.0, float(safety_drop_points or 0.0))
    if safety_drop >= 20.0 or exact_drop >= 20.0:
        severity = max(severity, 3)
        reasons.append(
            f"The forecast trajectory indicates a large reduction of "
            f"{max(safety_drop, exact_drop):.1f} points."
        )
    elif safety_drop >= 10.0 or exact_drop >= 10.0:
        severity = max(severity, 2)
        reasons.append(
            f"The forecast trajectory indicates a meaningful reduction of "
            f"{max(safety_drop, exact_drop):.1f} points."
        )

    if bhsi < 40.0:
        severity = max(severity, 2)
        reasons.append("BHSI indicates low six-hour environmental stability.")
    elif bhsi < 70.0:
        severity = max(severity, 1)
        reasons.append("BHSI indicates moderate six-hour environmental stability.")

    if rod_points_per_hour < -3.0:
        severity = max(severity, 2)
        reasons.append("RoD indicates rapid recent deterioration.")
    elif rod_points_per_hour < -0.5:
        severity = max(severity, 1)
        reasons.append("RoD indicates a slowly declining recent trend.")

    domain_shift_warnings = domain_shift_warnings or []
    if domain_shift_warnings:
        severity = max(severity, 1)
        reasons.append("One or more live inputs differ from the historical training domain.")
    if not history_sufficient:
        severity = max(severity, 1)
        reasons.append("The live history is shorter than the recommended 72 hours.")

    level = {0: "Excellent", 1: "Good", 2: "Poor", 3: "Critical"}[severity]
    actions = {
        "Excellent": ["Continue routine monitoring and verify sensor freshness."],
        "Good": [
            "Review the next readings and check any sensor moving away from its usual range."
        ],
      "Poor": [
    (
        "Inspect the colony soon; verify brood temperature, humidity, ventilation, "
        "CO₂ trend, relative weight change and sensor calibration."
    )
],
        "Critical": [
            "Perform an immediate physical hive inspection and confirm brood condition directly."
        ],
    }[level]

    if not reasons:
        reasons.append(
            "Current score, exact forecast, safety minimum, BHSI and RoD remain within "
            "the configured operating range."
        )

    return {
        "level": level,
        "title": f"{level} brood-health warning",
        "summary": (
            f"Current {current_condition_score:.1f}/100; exact forecast "
            f"{exact_forecast_score:.1f}/100; predicted safety minimum "
            f"{safety_minimum_score:.1f}/100."
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
    "compute_condition_history",
]
