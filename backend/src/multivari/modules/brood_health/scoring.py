from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

HEALTH_LEVEL_ORDER = ("Critical", "Poor", "Good", "Excellent")
LEVEL_TO_CODE = {name: index for index, name in enumerate(HEALTH_LEVEL_ORDER)}
CODE_TO_LEVEL = {index: name for name, index in LEVEL_TO_CODE.items()}


@dataclass(frozen=True)
class BroodHealthScoreConfig:
    """Transparent 1–100 brood-health condition index configuration.

    The common historical dataset contains a binary healthy/unhealthy label, not a
    measured continuous biological score. The 1–100 Brood Health Score is therefore
    an explicit research index built from sensor suitability. It must be presented as
    decision support and confirmed by physical brood inspection.
    """

    temperature_centre: float = 35.0
    temperature_scale: float = 4.0
    humidity_centre: float = 65.0
    humidity_scale: float = 16.0
    co2_good_max: float = 3_000.0
    co2_warning_max: float = 10_000.0
    co2_critical_max: float = 30_000.0
    temperature_weight: float = 0.45
    humidity_weight: float = 0.25
    co2_weight: float = 0.20
    weight_stability_weight: float = 0.10
    weight_reference_hours: int = 24
    weight_penalty_per_percentage_point: float = 8.0
    minimum_score: float = 1.0
    maximum_score: float = 100.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> BroodHealthScoreConfig:
        if not payload:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in payload.items() if key in allowed})


def classify_health_level(score: float) -> str:
    value = float(np.clip(score, 1.0, 100.0))
    if value < 40.0:
        return "Critical"
    if value < 60.0:
        return "Poor"
    if value < 80.0:
        return "Good"
    return "Excellent"


def health_level_code(score: float | pd.Series | np.ndarray) -> int | np.ndarray:
    if np.isscalar(score):
        return LEVEL_TO_CODE[classify_health_level(float(score))]
    values = np.asarray(score, dtype=float)
    return np.digitize(values, bins=[40.0, 60.0, 80.0], right=False).astype(int)


def _gaussian_suitability(values: pd.Series, centre: float, scale: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    z = (numeric - centre) / max(float(scale), 1e-6)
    return (100.0 * np.exp(-(z**2))).clip(0.0, 100.0)


def _co2_suitability(values: pd.Series, config: BroodHealthScoreConfig) -> pd.Series:
    co2 = pd.to_numeric(values, errors="coerce")
    result = np.select(
        [
            co2 <= config.co2_good_max,
            co2 <= config.co2_warning_max,
            co2 <= config.co2_critical_max,
        ],
        [
            100.0,
            100.0
            - (co2 - config.co2_good_max)
            * (35.0 / max(config.co2_warning_max - config.co2_good_max, 1.0)),
            65.0
            - (co2 - config.co2_warning_max)
            * (65.0 / max(config.co2_critical_max - config.co2_warning_max, 1.0)),
        ],
        default=0.0,
    )
    return pd.Series(result, index=values.index, dtype=float).clip(0.0, 100.0)


def compute_score_components(
    frame: pd.DataFrame,
    *,
    config: BroodHealthScoreConfig | None = None,
) -> pd.DataFrame:
    """Add causal sensor-component scores and the 1–100 Brood Health Score.

    Weight contributes through *stability* rather than absolute hive weight. This
    avoids turning hive size into a proxy for health and transfers better to unseen
    Sri Lankan hives.
    """

    cfg = config or BroodHealthScoreConfig()
    required = {"hive_id", "timestamp", "temperature_c", "humidity_pct", "co2_ppm", "weight_kg"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Brood Health Score input is missing columns: {missing}")

    out = frame.copy()
    out["temperature_component"] = _gaussian_suitability(
        out["temperature_c"], cfg.temperature_centre, cfg.temperature_scale
    )
    out["humidity_component"] = _gaussian_suitability(
        out["humidity_pct"], cfg.humidity_centre, cfg.humidity_scale
    )
    out["co2_component"] = _co2_suitability(out["co2_ppm"], cfg)

    weight = pd.to_numeric(out["weight_kg"], errors="coerce")
    previous = weight.groupby(out["hive_id"], sort=False).shift(cfg.weight_reference_hours)
    weight_change_pct = ((weight - previous) / previous.abs().clip(lower=1.0)) * 100.0
    out["weight_change_pct_24h"] = weight_change_pct
    out["weight_component"] = (
        100.0 - weight_change_pct.abs() * cfg.weight_penalty_per_percentage_point
    ).clip(0.0, 100.0)
    # Early rows do not yet have 24 h history. A neutral value prevents the missing
    # history from creating an artificial healthy or unhealthy classification.
    out["weight_component"] = out["weight_component"].fillna(70.0)

    score = (
        cfg.temperature_weight * out["temperature_component"]
        + cfg.humidity_weight * out["humidity_component"]
        + cfg.co2_weight * out["co2_component"]
        + cfg.weight_stability_weight * out["weight_component"]
    )
    out["brood_health_score"] = score.clip(cfg.minimum_score, cfg.maximum_score)
    out["condition_score"] = out["brood_health_score"]  # Backward-compatible API name.
    out["brood_health_level"] = out["brood_health_score"].map(classify_health_level)
    out["condition_level"] = out["brood_health_level"]
    return out


def score_definition(config: BroodHealthScoreConfig | None = None) -> dict[str, Any]:
    cfg = config or BroodHealthScoreConfig()
    return {
        "name": "Brood Health Score",
        "range": "1–100",
        "nature": "transparent sensor-derived research index",
        "components": [
            {"name": "Temperature suitability", "weight": cfg.temperature_weight},
            {"name": "Humidity suitability", "weight": cfg.humidity_weight},
            {"name": "CO₂ suitability", "weight": cfg.co2_weight},
            {"name": "24-hour weight stability", "weight": cfg.weight_stability_weight},
        ],
        "levels": [
            {"level": "Critical", "rule": "1 ≤ score < 40"},
            {"level": "Poor", "rule": "40 ≤ score < 60"},
            {"level": "Good", "rule": "60 ≤ score < 80"},
            {"level": "Excellent", "rule": "80 ≤ score ≤ 100"},
        ],
        "config": cfg.to_dict(),
        "limitation": (
            "The score is not a direct veterinary measurement. It is an interpretable "
            "sensor index and must be validated against physical brood inspections."
        ),
    }