from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

HEALTH_LEVEL_ORDER = ("Critical", "Poor", "Good", "Excellent")
LEVEL_TO_CODE = {name: index for index, name in enumerate(HEALTH_LEVEL_ORDER)}
CODE_TO_LEVEL = {index: name for name, index in LEVEL_TO_CODE.items()}

HEALTH_LEVEL_RULES = (
    {"level": "Critical", "minimum": 1.0, "maximum": 40.0, "rule": "1 ≤ score < 40"},
    {"level": "Poor", "minimum": 40.0, "maximum": 60.0, "rule": "40 ≤ score < 60"},
    {"level": "Good", "minimum": 60.0, "maximum": 80.0, "rule": "60 ≤ score < 80"},
    {"level": "Excellent", "minimum": 80.0, "maximum": 100.0, "rule": "80 ≤ score ≤ 100"},
)


@dataclass(frozen=True)
class BroodHealthScoreConfig:
    """Configuration for the transparent 1–100 brood-health research index.

    Centres and suitability functions are explicit assumptions. The four component
    weights are calibrated only on training hives, starting from the literature-informed
    prior below. They are not universal biological percentages.
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
    missing_weight_neutral_score: float = 70.0

    minimum_score: float = 1.0
    maximum_score: float = 100.0

    def validate(self) -> BroodHealthScoreConfig:
        weights = self.weights
        if any(value < 0 for value in weights.values()):
            raise ValueError("Brood-health score weights must be non-negative")
        if not np.isclose(sum(weights.values()), 1.0, atol=1e-6):
            raise ValueError(f"Brood-health score weights must sum to 1.0; received {weights}")
        if self.temperature_scale <= 0 or self.humidity_scale <= 0:
            raise ValueError("Suitability scales must be positive")
        return self

    @property
    def weights(self) -> dict[str, float]:
        return {
            "temperature": float(self.temperature_weight),
            "humidity": float(self.humidity_weight),
            "co2": float(self.co2_weight),
            "weight_stability": float(self.weight_stability_weight),
        }

    def with_weights(
        self,
        *,
        temperature: float,
        humidity: float,
        co2: float,
        weight_stability: float,
    ) -> BroodHealthScoreConfig:
        return replace(
            self,
            temperature_weight=float(temperature),
            humidity_weight=float(humidity),
            co2_weight=float(co2),
            weight_stability_weight=float(weight_stability),
        ).validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> BroodHealthScoreConfig:
        if not payload:
            return cls().validate()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in payload.items() if key in allowed}).validate()


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


def compute_unweighted_components(
    frame: pd.DataFrame,
    *,
    config: BroodHealthScoreConfig | None = None,
) -> pd.DataFrame:
    """Calculate the four component scores without combining their weights."""

    cfg = (config or BroodHealthScoreConfig()).validate()
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
    change_pct = ((weight - previous) / previous.abs().clip(lower=1.0)) * 100.0
    out["weight_change_pct_24h"] = change_pct
    out["weight_component"] = (
        100.0 - change_pct.abs() * cfg.weight_penalty_per_percentage_point
    ).clip(0.0, 100.0)
    out["weight_component"] = out["weight_component"].fillna(cfg.missing_weight_neutral_score)
    return out


def combine_components(
    components: pd.DataFrame,
    *,
    config: BroodHealthScoreConfig | None = None,
) -> pd.Series:
    cfg = (config or BroodHealthScoreConfig()).validate()
    score = (
        cfg.temperature_weight * components["temperature_component"]
        + cfg.humidity_weight * components["humidity_component"]
        + cfg.co2_weight * components["co2_component"]
        + cfg.weight_stability_weight * components["weight_component"]
    )
    return score.clip(cfg.minimum_score, cfg.maximum_score)


def compute_score_components(
    frame: pd.DataFrame,
    *,
    config: BroodHealthScoreConfig | None = None,
) -> pd.DataFrame:
    """Add component scores, the 1–100 score and non-overlapping health levels."""

    cfg = (config or BroodHealthScoreConfig()).validate()
    out = compute_unweighted_components(frame, config=cfg)
    out["brood_health_score"] = combine_components(out, config=cfg)
    out["condition_score"] = out["brood_health_score"]
    out["brood_health_level"] = out["brood_health_score"].map(classify_health_level)
    out["condition_level"] = out["brood_health_level"]
    return out


def score_definition(config: BroodHealthScoreConfig | None = None) -> dict[str, Any]:
    cfg = (config or BroodHealthScoreConfig()).validate()
    return {
        "name": "Brood Health Score",
        "range": "1–100",
        "nature": "transparent sensor-derived research index",
        "components": [
            {
                "name": "Temperature suitability",
                "key": "temperature",
                "weight": cfg.temperature_weight,
            },
            {
                "name": "Humidity suitability",
                "key": "humidity",
                "weight": cfg.humidity_weight,
            },
            {"name": "CO₂ suitability", "key": "co2", "weight": cfg.co2_weight},
            {
                "name": "24-hour relative weight stability",
                "key": "weight_stability",
                "weight": cfg.weight_stability_weight,
            },
        ],
        "levels": list(HEALTH_LEVEL_RULES),
        "config": cfg.to_dict(),
        "weight_interpretation": (
            "The values are calibrated on training hives under biological ordering "
            "constraints. They express the contribution to this research index, not "
            "universal biological percentages."
        ),
        "weight_transfer_strategy": (
            "Only relative weight change and stability are used. Absolute hive weight "
            "is excluded from the forecasting features to reduce European/Sri Lankan "
            "scale and hive-size mismatch."
        ),
        "limitation": (
            "The score is a non-invasive sensor index and is not a direct veterinary "
            "measurement. Poor and Critical results require physical brood inspection."
        ),
    }
