import numpy as np
import pandas as pd

from multivari.modules.harvesting.provisional_hui_regression import (
    _regression_metrics,
    add_future_hui_target,
    assign_provisional_hui_class,
    build_current_provisional_hui,
    evaluate_hui_research_gate,
)


def _component_config() -> dict[str, dict[str, object]]:
    return {
        "weight_position": {
            "column": "weight_relative_to_max_168h",
            "direction": "higher",
            "weight": 0.30,
        },
        "recent_accumulation": {
            "column": "weight_delta_72h_kg",
            "direction": "higher",
            "weight": 0.25,
            "clip_lower": 0.0,
        },
        "weight_plateau": {
            "column": "weight_trend_72h_kg_per_hour",
            "direction": "lower_absolute",
            "weight": 0.25,
        },
        "environmental_stability": {
            "column": "environmental_variability_72h",
            "direction": "lower",
            "weight": 0.10,
        },
        "temperature_stability": {
            "column": "temperature_c_range_24h",
            "direction": "lower",
            "weight": 0.10,
        },
    }


def test_hui_classes_follow_fixed_boundaries() -> None:
    values = pd.Series([0.0, 39.9, 40.0, 59.9, 60.0, 79.9, 80.0])
    classes = assign_provisional_hui_class(
        values,
        not_ready_upper=40.0,
        approaching_upper=60.0,
        ready_upper=80.0,
    )

    assert classes.tolist() == [
        "Not Ready",
        "Not Ready",
        "Approaching Harvest",
        "Approaching Harvest",
        "Ready — Inspection Recommended",
        "Ready — Inspection Recommended",
        "High-Priority Harvest Review",
    ]


def test_current_hui_is_bounded_and_uses_train_normalization() -> None:
    rows = 20
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=rows,
                freq="h",
            ),
            "hive_id": ["h1"] * rows,
            "split": ["train"] * 15 + ["validation"] * 5,
            "weight_relative_to_max_168h": np.linspace(
                0.70,
                1.00,
                rows,
            ),
            "weight_delta_72h_kg": np.linspace(0.0, 5.0, rows),
            "weight_trend_72h_kg_per_hour": np.linspace(
                0.2,
                0.0,
                rows,
            ),
            "environmental_variability_72h": np.linspace(
                3.0,
                0.2,
                rows,
            ),
            "temperature_c_range_24h": np.linspace(
                5.0,
                0.5,
                rows,
            ),
            "co2_flatline_24h_1": [0.0] * rows,
            "co2_flatline_72h_1": [0.0] * rows,
        }
    )

    result, parameters = build_current_provisional_hui(
        frame,
        component_config=_component_config(),
        lower_quantile=0.10,
        upper_quantile=0.90,
        quality_columns=[
            "co2_flatline_24h_1",
            "co2_flatline_72h_1",
        ],
        quality_penalty_per_flag=0.25,
        minimum_quality_factor=0.50,
        class_config={
            "not_ready_upper": 40.0,
            "approaching_upper": 60.0,
            "ready_upper": 80.0,
        },
    )

    assert result["provisional_hui"].between(0.0, 100.0).all()
    assert parameters["normalization_source_split"] == "train"
    assert result["provisional_hui_class"].notna().all()


def test_quality_flags_reduce_hui() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=4,
                freq="h",
            ),
            "hive_id": ["h1"] * 4,
            "split": ["train"] * 4,
            "weight_relative_to_max_168h": [0.95] * 4,
            "weight_delta_72h_kg": [2.0] * 4,
            "weight_trend_72h_kg_per_hour": [0.01] * 4,
            "environmental_variability_72h": [1.0] * 4,
            "temperature_c_range_24h": [2.0] * 4,
            "co2_flatline_24h_1": [0.0, 1.0, 0.0, 1.0],
            "co2_flatline_72h_1": [0.0, 0.0, 1.0, 1.0],
        }
    )

    result, _ = build_current_provisional_hui(
        frame,
        component_config=_component_config(),
        lower_quantile=0.10,
        upper_quantile=0.90,
        quality_columns=[
            "co2_flatline_24h_1",
            "co2_flatline_72h_1",
        ],
        quality_penalty_per_flag=0.25,
        minimum_quality_factor=0.50,
        class_config={
            "not_ready_upper": 40.0,
            "approaching_upper": 60.0,
            "ready_upper": 80.0,
        },
    )

    assert result.loc[0, "hui_data_quality_factor"] == 1.0
    assert result.loc[3, "hui_data_quality_factor"] == 0.5
    assert result.loc[3, "provisional_hui"] <= result.loc[0, "provisional_hui"]


def test_future_hui_target_does_not_cross_split() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-02 00:00",
                ]
            ),
            "hive_id": ["h1", "h1"],
            "split": ["train", "validation"],
            "provisional_hui": [20.0, 80.0],
        }
    )

    result = add_future_hui_target(
        frame,
        horizon_hours=24,
    )

    assert pd.isna(
        result.loc[
            0,
            "future_provisional_hui_24h",
        ]
    )


def test_regression_metrics_report_point_tolerances() -> None:
    metrics = _regression_metrics(
        np.array([10.0, 20.0, 30.0]),
        np.array([12.0, 26.0, 39.0]),
    )

    assert metrics["within_5_points_fraction"] == 1 / 3
    assert metrics["within_10_points_fraction"] == 1.0
    assert metrics["bias"] > 0


def test_hui_gate_requires_multiple_improved_horizons() -> None:
    comparison = pd.DataFrame(
        {
            "horizon_hours": [24, 48, 72],
            "model": [
                "persistence",
                "persistence",
                "persistence",
            ],
            "status": ["ok", "ok", "ok"],
            "validation_mae": [10.0, 10.0, 10.0],
        }
    )
    summary = {
        "horizons": {
            "24": {
                "selected_model": "ridge",
                "selected_feature_set": "weight_only",
                "validation": {"mae": 8.0},
                "test": {"mae": 9.0},
            },
            "48": {
                "selected_model": "ridge",
                "selected_feature_set": "weight_only",
                "validation": {"mae": 8.5},
                "test": {"mae": 9.0},
            },
            "72": {
                "selected_model": "persistence",
                "selected_feature_set": "persistence",
                "validation": {"mae": 10.0},
                "test": {"mae": 10.5},
            },
        }
    }

    gate = evaluate_hui_research_gate(
        comparison,
        summary,
        horizons_hours=[24, 48, 72],
        minimum_improvement=0.05,
        required_improved_horizons=2,
        maximum_test_to_validation_ratio=2.0,
    )

    assert gate["gate_passed"]
    assert gate["improved_horizon_count"] == 2
    assert not gate["ready_for_operational_hui"]
