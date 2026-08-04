from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AbscondingSettings:
    # Module-specific dataset. The other BeeHive modules continue using common_clean.parquet.
    data_input_path: str = "data/raw/absconding/hive_data_with_features.csv"
    data_clean_path: str = "data/processed/absconding_clean.parquet"
    data_manifest_path: str = "data/manifests/absconding_split_manifest.parquet"
    data_profile_path: str = "artifacts/reports/absconding/absconding_data_profile.json"

    source_timestamp_column: str = "timestamp"
    source_hive_column: str = "hive_id"
    source_temperature_column: str = "internal_temperature_c"
    source_humidity_column: str = "internal_humidity_pct"
    source_co2_column: str = "co2_ppm"
    source_weight_column: str = "hive_weight_kg"
    source_external_temperature_column: str = "external_temperature_c"
    source_external_humidity_column: str = "external_humidity_pct"
    source_event_column: str = "absconding_event_label"
    source_precomputed_target_column: str = "absconding_label_next_72h"
    interpolation_limit_hours: int = 6

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    boundary_gap_hours: int = 72

    event_column: str = "absconding_happened_1"
    active_event_column: str = "absconding_active_1"
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
    final_training_rows: int = 100_000
    anomaly_training_rows: int = 50_000
    permutation_importance_rows: int = 3_000
    timeline_points_per_hive: int = 336
    medium_threshold_ratio: float = 0.60
    arm_change_hours: int = 24
    arm_escalation_threshold: float = 0.10

    @classmethod
    def from_yaml(cls, path: str | Path) -> AbscondingSettings:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data = payload.get("data", {})
        columns = data.get("columns", {})
        split = payload.get("split", {})
        module = payload.get("module", {})
        features = payload.get("features", {})
        training = payload.get("training", {})
        risk = payload.get("risk", {})

        values: dict[str, Any] = {
            **module,
            **training,
            **risk,
            "data_input_path": data.get("input_path", cls.data_input_path),
            "data_clean_path": data.get("clean_path", cls.data_clean_path),
            "data_manifest_path": data.get("manifest_path", cls.data_manifest_path),
            "data_profile_path": data.get("profile_path", cls.data_profile_path),
            "source_timestamp_column": columns.get("timestamp", cls.source_timestamp_column),
            "source_hive_column": columns.get("hive", cls.source_hive_column),
            "source_temperature_column": columns.get("temperature", cls.source_temperature_column),
            "source_humidity_column": columns.get("humidity", cls.source_humidity_column),
            "source_co2_column": columns.get("co2", cls.source_co2_column),
            "source_weight_column": columns.get("weight", cls.source_weight_column),
            "source_external_temperature_column": columns.get(
                "external_temperature", cls.source_external_temperature_column
            ),
            "source_external_humidity_column": columns.get(
                "external_humidity", cls.source_external_humidity_column
            ),
            "source_event_column": columns.get("event", cls.source_event_column),
            "source_precomputed_target_column": columns.get(
                "precomputed_72h_target", cls.source_precomputed_target_column
            ),
            "interpolation_limit_hours": data.get(
                "interpolation_limit_hours", cls.interpolation_limit_hours
            ),
            "train_fraction": split.get("train_fraction", cls.train_fraction),
            "validation_fraction": split.get("validation_fraction", cls.validation_fraction),
            "boundary_gap_hours": split.get("boundary_gap_hours", cls.boundary_gap_hours),
            "lags_hours": tuple(features.get("lags_hours", cls.lags_hours)),
            "change_hours": tuple(features.get("change_hours", cls.change_hours)),
            "rolling_windows_hours": tuple(
                features.get("rolling_windows_hours", cls.rolling_windows_hours)
            ),
            "rolling_statistics": tuple(features.get("rolling_statistics", cls.rolling_statistics)),
            "model_candidates": tuple(training.get("model_candidates", cls.model_candidates)),
        }
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in self.__dict__.items()
        }
