from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AbscondingSettings:
    event_column: str = "absconding_happened_1"
    target_column: str = "absconding_within_24h"
    prediction_horizon_hours: int = 24
    event_merge_gap_hours: int = 24
    minimum_history_hours: int = 168
    lags_hours: tuple[int, ...] = (1, 6, 24, 72, 168)
    change_hours: tuple[int, ...] = (1, 6, 24, 72, 168)
    rolling_windows_hours: tuple[int, ...] = (6, 24, 72, 168)
    rolling_statistics: tuple[str, ...] = ("mean", "std", "min", "max")
    random_state: int = 42
    model_candidates: tuple[str, ...] = (
        "dummy_prior",
        "rule_based_stress",
        "gaussian_nb",
        "logistic_balanced",
        "ridge_classifier",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "isolation_forest",
    )
    threshold_beta: float = 2.0
    maximum_validation_alert_fraction: float = 0.05
    comparison_training_rows: int = 60_000
    final_training_rows: int = 80_000
    anomaly_training_rows: int = 50_000
    permutation_importance_rows: int = 3_000
    timeline_points_per_hive: int = 336
    medium_threshold_ratio: float = 0.60
    arm_change_hours: int = 24
    arm_escalation_threshold: float = 0.10

    @classmethod
    def from_yaml(cls, path: str | Path) -> AbscondingSettings:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        module = payload.get("module", {})
        features = payload.get("features", {})
        training = payload.get("training", {})
        risk = payload.get("risk", {})

        values: dict[str, Any] = {
            **module,
            **training,
            **risk,
            "lags_hours": tuple(features.get("lags_hours", cls.lags_hours)),
            "change_hours": tuple(features.get("change_hours", cls.change_hours)),
            "rolling_windows_hours": tuple(
                features.get("rolling_windows_hours", cls.rolling_windows_hours)
            ),
            "rolling_statistics": tuple(
                features.get("rolling_statistics", cls.rolling_statistics)
            ),
            "model_candidates": tuple(
                training.get("model_candidates", cls.model_candidates)
            ),
        }
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in self.__dict__.items()
        }
