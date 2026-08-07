from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .features import TARGET_COLUMN
from .scoring import (
    BroodHealthScoreConfig,
    combine_components,
    compute_unweighted_components,
)


@dataclass(frozen=True)
class WeightCalibrationResult:
    config: BroodHealthScoreConfig
    comparison: pd.DataFrame
    method: dict[str, Any]


def _candidate_weights() -> list[tuple[float, float, float, float]]:
    """Generate interpretable candidates with the required biological ordering.

    Temperature >= humidity >= CO2 >= relative weight stability.
    """

    values: list[tuple[float, float, float, float]] = []
    for temperature, humidity, co2 in product(
        (0.35, 0.40, 0.45, 0.50),
        (0.20, 0.25, 0.30),
        (0.15, 0.20, 0.25),
    ):
        weight = round(1.0 - temperature - humidity - co2, 10)
        if not 0.05 <= weight <= 0.20:
            continue
        if not (temperature >= humidity >= co2 >= weight):
            continue
        values.append((temperature, humidity, co2, weight))
    prior = (0.45, 0.25, 0.20, 0.10)
    if prior not in values:
        values.append(prior)
    return sorted(set(values))


def calibrate_component_weights(
    frame: pd.DataFrame,
    *,
    training_hives: set[str],
    prior_config: BroodHealthScoreConfig | None = None,
) -> WeightCalibrationResult:
    """Calibrate weights using only the historical training hives.

    The observed healthy/unhealthy label is used only to check whether a candidate
    composite index aligns with the supplied health status. It is never used as a
    forecasting feature. A small penalty keeps the selected weights close to the
    literature-informed prior unless the training data provide a meaningful benefit.
    """

    cfg = (prior_config or BroodHealthScoreConfig()).validate()
    if TARGET_COLUMN not in frame.columns:
        return WeightCalibrationResult(
            config=cfg,
            comparison=pd.DataFrame(),
            method={
                "status": "not_run",
                "reason": f"{TARGET_COLUMN} was unavailable",
                "selected_weights": cfg.weights,
            },
        )

    selected = frame.loc[frame["hive_id"].astype(str).isin(training_hives)].copy()
    selected[TARGET_COLUMN] = pd.to_numeric(selected[TARGET_COLUMN], errors="coerce")
    selected = selected.loc[selected[TARGET_COLUMN].isin([0, 1])].copy()
    if selected.empty or selected[TARGET_COLUMN].nunique() < 2:
        return WeightCalibrationResult(
            config=cfg,
            comparison=pd.DataFrame(),
            method={
                "status": "not_run",
                "reason": "Training hives did not contain both observed health classes",
                "selected_weights": cfg.weights,
            },
        )

    components = compute_unweighted_components(selected, config=cfg)
    observed = selected[TARGET_COLUMN].astype(int).to_numpy()
    prior = np.array(
        [
            cfg.temperature_weight,
            cfg.humidity_weight,
            cfg.co2_weight,
            cfg.weight_stability_weight,
        ],
        dtype=float,
    )

    rows: list[dict[str, Any]] = []
    for temperature, humidity, co2, weight_stability in _candidate_weights():
        candidate = cfg.with_weights(
            temperature=temperature,
            humidity=humidity,
            co2=co2,
            weight_stability=weight_stability,
        )
        score = combine_components(components, config=candidate).to_numpy(dtype=float)
        predicted = (score >= 60.0).astype(int)

        balanced = float(balanced_accuracy_score(observed, predicted))
        macro_f1 = float(f1_score(observed, predicted, average="macro", zero_division=0))
        accuracy = float(accuracy_score(observed, predicted))
        distance = float(
            np.abs(
                np.array([temperature, humidity, co2, weight_stability], dtype=float)
                - prior
            ).sum()
        )
        # The objective favours class-balanced alignment while discouraging arbitrary
        # movement away from the prior biological ordering.
        objective = 0.65 * balanced + 0.25 * macro_f1 + 0.10 * accuracy - 0.08 * distance
        level_codes = np.digitize(score, bins=[40.0, 60.0, 80.0], right=False)
        rows.append(
            {
                "temperature_weight": temperature,
                "humidity_weight": humidity,
                "co2_weight": co2,
                "weight_stability_weight": weight_stability,
                "balanced_accuracy": balanced,
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "distance_from_prior": distance,
                "objective": objective,
                "critical_rate": float((level_codes == 0).mean()),
                "poor_rate": float((level_codes == 1).mean()),
                "good_rate": float((level_codes == 2).mean()),
                "excellent_rate": float((level_codes == 3).mean()),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["objective", "balanced_accuracy", "macro_f1"],
        ascending=False,
    ).reset_index(drop=True)
    best = comparison.iloc[0]
    selected_config = cfg.with_weights(
        temperature=float(best["temperature_weight"]),
        humidity=float(best["humidity_weight"]),
        co2=float(best["co2_weight"]),
        weight_stability=float(best["weight_stability_weight"]),
    )
    return WeightCalibrationResult(
        config=selected_config,
        comparison=comparison,
        method={
            "status": "completed",
            "scope": "training hives only",
            "candidate_count": len(comparison),
            "selection_metric": (
                "0.65 balanced accuracy + 0.25 macro F1 + 0.10 accuracy "
                "- 0.08 L1 distance from the prior weights"
            ),
            "health_alignment_threshold": 60.0,
            "ordering_constraint": (
                "temperature >= humidity >= CO2 >= relative weight stability"
            ),
            "prior_weights": cfg.weights,
            "selected_weights": selected_config.weights,
            "interpretation": (
                "The calibration validates the contribution of the four available "
                "sensor components within this dataset. It does not prove a universal "
                "biological percentage."
            ),
        },
    )
