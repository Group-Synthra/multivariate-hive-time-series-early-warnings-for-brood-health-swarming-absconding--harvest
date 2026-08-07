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


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _beekeeper_actions(
    *,
    severity: str,
    forecast_bhsi: float,
    forecast_rod_points_per_hour: float,
    current_temperature_c: float | None,
    current_humidity_pct: float | None,
    current_co2_ppm: float | None,
    weight_change_pct_24h: float | None,
    weight_component: float | None,
) -> list[str]:
    """Return concise, conditional actions suitable for the live admin dashboard.

    Sensor values are used to prioritise checks, not to diagnose disease. Feeding is
    recommended only after the beekeeper confirms that food stores are genuinely low.
    """

    actions: list[str] = []

    if severity == "Normal":
        _append_unique(
            actions,
            "Continue routine monitoring and inspect on the normal colony-management schedule.",
        )
    elif severity == "Watch":
        _append_unique(
            actions,
            "Review the next 2–3 IoT readings; inspect during the next suitable daylight period if the decline or instability persists.",
        )
        _append_unique(
            actions,
            "Verify sensor placement, hive-scale tare and entrance condition before making a management change.",
        )
    elif severity == "Warning":
        _append_unique(
            actions,
            "Inspect the brood nest within 6–12 hours: check brood pattern, eggs/queen-right status, food stores, entrance airflow and abnormal brood or odour.",
        )
        _append_unique(
            actions,
            "Check Varroa with a validated monitoring method and select treatment only from the measured mite level and locally approved guidance.",
        )
    else:  # Critical Alert
        _append_unique(
            actions,
            "Inspect the colony as soon as conditions are safe; confirm brood condition, queen status, food stores, overheating/chilling and pest or disease signs.",
        )
        _append_unique(
            actions,
            "If serious brood disease is suspected, avoid moving comb or equipment between colonies and contact the relevant local apiculture or veterinary authority.",
        )

    if current_temperature_c is not None:
        if current_temperature_c > 37.0:
            _append_unique(
                actions,
                "Check shade, water availability and entrance/ventilation for obstruction; correct confirmed heat stress without unnecessarily exposing brood.",
            )
        elif current_temperature_c < 32.0:
            _append_unique(
                actions,
                "Check colony strength, brood coverage and drafts; minimise prolonged brood-nest opening and correct confirmed cold exposure.",
            )

    if current_humidity_pct is not None:
        if current_humidity_pct > 80.0:
            _append_unique(
                actions,
                "Inspect for condensation, leaks and restricted airflow; correct confirmed excess moisture while keeping the brood nest protected.",
            )
        elif current_humidity_pct < 45.0:
            _append_unique(
                actions,
                "Check water availability, excessive airflow and humidity-sensor position before changing ventilation.",
            )

    if current_co2_ppm is not None and current_co2_ppm > 5_000.0:
        _append_unique(
            actions,
            "Check that the entrance and ventilation paths are not blocked and verify the CO₂ sensor position/calibration.",
        )

    weight_signal_low = (
        (weight_change_pct_24h is not None and weight_change_pct_24h <= -3.0)
        or (weight_component is not None and weight_component < 50.0)
    )
    if weight_signal_low:
        _append_unique(
            actions,
            "Verify the scale/tare and inspect honey or nectar stores. If stores are genuinely low and forage is inadequate, provide clean sucrose syrup in an internal feeder according to local seasonal practice; avoid open feeding and do not use feeding as a disease treatment.",
        )

    if forecast_bhsi < 40.0:
        _append_unique(
            actions,
            "Because forecast stability is low, recheck temperature, moisture, airflow and sensor consistency over the next readings.",
        )

    if forecast_rod_points_per_hour < -0.5 and severity in {"Watch", "Warning"}:
        _append_unique(
            actions,
            "Compare brood pattern, queen activity and food stores with the previous inspection to identify the cause of the declining trend.",
        )

    return actions[:5]


def build_warning_payload(
    *,
    exact_forecast_score: float,
    safety_minimum_score: float,
    current_condition_score: float,
    forecast_bhsi: float | None = None,
    forecast_rod_points_per_hour: float | None = None,
    bhsi: float | None = None,
    rod_points_per_hour: float | None = None,
    exact_forecast_drop_points: float | None = None,
    safety_drop_points: float | None = None,
    domain_shift_warnings: list[str] | None = None,
    history_sufficient: bool = True,
    current_temperature_c: float | None = None,
    current_humidity_pct: float | None = None,
    current_co2_ppm: float | None = None,
    weight_change_pct_24h: float | None = None,
    weight_component: float | None = None,
) -> dict[str, Any]:
    """Build a composite deterioration alert without reusing health-level labels.

    Health levels (Critical/Poor/Good/Excellent) describe individual score values.
    Alert severities (Normal/Watch/Warning/Critical Alert) combine the current score,
    exact +6-hour score, safety minimum, forecast decline, Forecast BHSI and Forecast
    RoD. Data-quality notes affect confidence but do not independently escalate health.
    """

    current_score = float(np.clip(current_condition_score, 1.0, 100.0))
    exact_score = float(np.clip(exact_forecast_score, 1.0, 100.0))
    minimum_score = float(np.clip(safety_minimum_score, 1.0, 100.0))
    forecast_bhsi = float(
        np.clip(
            forecast_bhsi
            if forecast_bhsi is not None
            else (bhsi if bhsi is not None else 100.0),
            0.0,
            100.0,
        )
    )
    forecast_rod_points_per_hour = float(
        forecast_rod_points_per_hour
        if forecast_rod_points_per_hour is not None
        else (rod_points_per_hour if rod_points_per_hour is not None else 0.0)
    )

    current_level = classify_health_level(current_score)
    exact_level = classify_health_level(exact_score)
    minimum_level = classify_health_level(minimum_score)

    exact_drop = max(0.0, float(exact_forecast_drop_points or 0.0))
    safety_drop = max(0.0, float(safety_drop_points or 0.0))
    maximum_drop = max(exact_drop, safety_drop)

    score_levels = {current_level, exact_level, minimum_level}
    rapid_unstable_decline = (
        maximum_drop >= 20.0
        and (forecast_bhsi < 40.0 or forecast_rod_points_per_hour < -3.0)
    )

    if "Critical" in score_levels or rapid_unstable_decline:
        severity = "Critical Alert"
    elif (
        "Poor" in score_levels
        or maximum_drop >= 10.0
        or forecast_bhsi < 40.0
        or forecast_rod_points_per_hour < -3.0
    ):
        severity = "Warning"
    elif (
        maximum_drop >= 5.0
        or forecast_bhsi < 70.0
        or forecast_rod_points_per_hour < -0.5
    ):
        severity = "Watch"
    else:
        severity = "Normal"

    reasons: list[str] = []
    if current_level in {"Poor", "Critical"}:
        reasons.append(f"Current health is {current_level} ({current_score:.2f}/100).")
    if exact_level in {"Poor", "Critical"}:
        reasons.append(
            f"Exact +6-hour health is {exact_level} ({exact_score:.2f}/100)."
        )
    if (
        minimum_level in {"Poor", "Critical"}
        and minimum_score < exact_score - 0.005
    ):
        reasons.append(
            f"The predicted path reaches a {minimum_level} safety minimum of "
            f"{minimum_score:.2f}/100."
        )
    if maximum_drop >= 5.0:
        reasons.append(
            f"The predicted six-hour path drops by up to {maximum_drop:.2f} points."
        )
    if forecast_bhsi < 40.0:
        reasons.append(
            f"Forecast BHSI is {forecast_bhsi:.2f}/100 (Low stability)."
        )
    elif forecast_bhsi < 70.0:
        reasons.append(
            f"Forecast BHSI is {forecast_bhsi:.2f}/100 (Moderate stability)."
        )
    if forecast_rod_points_per_hour < -3.0:
        reasons.append(
            f"Forecast RoD is {forecast_rod_points_per_hour:.2f} points/hour "
            "(Rapid Declining)."
        )
    elif forecast_rod_points_per_hour < -0.5:
        reasons.append(
            f"Forecast RoD is {forecast_rod_points_per_hour:.2f} points/hour "
            "(Slow Declining)."
        )

    if not reasons:
        reasons.append(
            "The six-hour score path, Forecast BHSI and Forecast RoD remain within the configured monitoring range."
        )

    confidence_notes: list[str] = []
    for item in domain_shift_warnings or []:
        _append_unique(confidence_notes, str(item))
    if not history_sufficient:
        _append_unique(
            confidence_notes,
            "Live history is shorter than the recommended 72 hours; interpret the forecast with extra caution.",
        )

    urgency = {
        "Normal": "Routine monitoring",
        "Watch": "Review next readings",
        "Warning": "Inspect within 6–12 h",
        "Critical Alert": "Inspect as soon as safe",
    }[severity]
    title = {
        "Normal": "Conditions stable",
        "Watch": "Forecast watch",
        "Warning": "Deterioration warning",
        "Critical Alert": "Critical brood-health alert",
    }[severity]

    actions = _beekeeper_actions(
        severity=severity,
        forecast_bhsi=forecast_bhsi,
        forecast_rod_points_per_hour=forecast_rod_points_per_hour,
        current_temperature_c=_optional_float(current_temperature_c),
        current_humidity_pct=_optional_float(current_humidity_pct),
        current_co2_ppm=_optional_float(current_co2_ppm),
        weight_change_pct_24h=_optional_float(weight_change_pct_24h),
        weight_component=_optional_float(weight_component),
    )

    return {
        # ``level`` is retained for API compatibility; it is now alert severity,
        # not a Brood Health Score class.
        "level": severity,
        "severity": severity,
        "title": title,
        "urgency": urgency,
        "current_health_level": current_level,
        "predicted_health_level": exact_level,
        "safety_minimum_level": minimum_level,
        "summary": (
            f"Current {current_score:.2f}/100 ({current_level}) · "
            f"+6 h {exact_score:.2f}/100 ({exact_level}) · "
            f"Safety minimum {minimum_score:.2f}/100 ({minimum_level})."
        ),
        "reasons": reasons,
        "recommended_actions": actions,
        "confidence_notes": confidence_notes,
        "requires_physical_confirmation": True,
        "components": {
            "current_score": round(current_score, 2),
            "exact_score": round(exact_score, 2),
            "safety_minimum_score": round(minimum_score, 2),
            "maximum_drop_points": round(maximum_drop, 2),
            "forecast_bhsi": round(forecast_bhsi, 2),
            "forecast_rod_points_per_hour": round(
                forecast_rod_points_per_hour, 2
            ),
        },
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
