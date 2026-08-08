import numpy as np
import pandas as pd

from multivari.modules.harvesting.forecast_readiness import (
    assign_candidate_window,
    assign_readiness_class,
    build_provisional_readiness_scores,
    evaluate_forecasting_research_gate,
    scale_higher_better,
    scale_lower_better,
)


def test_forecasting_gate_passes_two_improved_horizons() -> None:
    comparison = pd.DataFrame(
        {
            "horizon_hours": [24, 48, 72],
            "model": [
                "persistence",
                "persistence",
                "persistence",
            ],
            "status": ["ok", "ok", "ok"],
            "validation_mae": [1.0, 2.0, 3.0],
        }
    )
    summary = {
        "horizons": {
            "24": {
                "selected_model": "ridge",
                "selected_feature_set": "weight_only",
                "validation": {"mae": 0.8},
                "test": {"mae": 1.0},
            },
            "48": {
                "selected_model": "xgboost",
                "selected_feature_set": "no_humidity",
                "validation": {"mae": 1.7},
                "test": {"mae": 2.0},
            },
            "72": {
                "selected_model": "lightgbm",
                "selected_feature_set": "no_humidity",
                "validation": {"mae": 2.9},
                "test": {"mae": 3.2},
            },
        }
    }

    result = evaluate_forecasting_research_gate(
        comparison,
        summary,
        horizons_hours=[24, 48, 72],
        minimum_validation_mae_improvement_fraction=0.02,
        required_improved_horizons=2,
        require_72h_not_worse_than_persistence=True,
        maximum_test_to_validation_mae_ratio=2.0,
    )

    assert result["ready_for_readiness_prototype"]
    assert result["improved_horizon_count"] == 3


def test_forecasting_gate_fails_when_persistence_is_not_beaten() -> None:
    comparison = pd.DataFrame(
        {
            "horizon_hours": [24, 48, 72],
            "model": [
                "persistence",
                "persistence",
                "persistence",
            ],
            "status": ["ok", "ok", "ok"],
            "validation_mae": [1.0, 2.0, 3.0],
        }
    )
    summary = {
        "horizons": {
            str(horizon): {
                "selected_model": "persistence",
                "selected_feature_set": "weight_only",
                "validation": {"mae": baseline},
                "test": {"mae": baseline},
            }
            for horizon, baseline in (
                (24, 1.0),
                (48, 2.0),
                (72, 3.0),
            )
        }
    }

    result = evaluate_forecasting_research_gate(
        comparison,
        summary,
        horizons_hours=[24, 48, 72],
        minimum_validation_mae_improvement_fraction=0.02,
        required_improved_horizons=2,
        require_72h_not_worse_than_persistence=True,
        maximum_test_to_validation_mae_ratio=2.0,
    )

    assert not result["ready_for_readiness_prototype"]
    assert result["improved_horizon_count"] == 0


def test_scalers_clip_to_unit_interval() -> None:
    values = pd.Series([-1.0, 0.0, 5.0, 10.0, 12.0])

    higher = scale_higher_better(
        values,
        lower=0.0,
        upper=10.0,
    )
    lower = scale_lower_better(
        values,
        lower=0.0,
        upper=10.0,
    )

    assert higher.between(0.0, 1.0).all()
    assert lower.between(0.0, 1.0).all()
    assert higher.iloc[0] == 0.0
    assert higher.iloc[-1] == 1.0
    assert lower.iloc[0] == 1.0
    assert lower.iloc[-1] == 0.0


def test_readiness_class_uses_monotonic_thresholds() -> None:
    score = pd.Series([10.0, 50.0, 70.0, 90.0])
    result = assign_readiness_class(
        score,
        {
            "approaching": 40.0,
            "ready": 60.0,
            "high_priority": 80.0,
        },
    )

    assert result.tolist() == [
        "Not Ready",
        "Approaching",
        "Ready",
        "High Priority",
    ]


def test_candidate_window_uses_earliest_plateau_horizon() -> None:
    frame = pd.DataFrame(
        {
            "readiness_class": [
                "Ready",
                "High Priority",
                "Not Ready",
            ],
            "predicted_rate_24h_kg_per_hour": [
                0.01,
                0.10,
                0.01,
            ],
            "predicted_rate_48h_kg_per_hour": [
                0.02,
                0.02,
                0.01,
            ],
            "predicted_rate_72h_kg_per_hour": [
                0.03,
                0.03,
                0.01,
            ],
        }
    )

    result = assign_candidate_window(
        frame,
        class_column="readiness_class",
        plateau_rate_thresholds={
            "24": 0.05,
            "48": 0.05,
            "72": 0.05,
        },
    )

    assert result.tolist() == [
        "0-24 hours",
        "24-48 hours",
        "No candidate window",
    ]


def test_provisional_score_is_finite_and_bounded() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=40,
        freq="h",
    )
    split = ["train"] * 30 + ["validation"] * 10
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * 40,
            "split": split,
            "weight_delta_72h_kg": np.linspace(0.0, 5.0, 40),
            "weight_distance_from_max_168h_kg": np.linspace(
                5.0,
                0.0,
                40,
            ),
            "temperature_c_std_24h": np.linspace(2.0, 0.2, 40),
            "co2_ppm_std_24h": np.linspace(100.0, 10.0, 40),
            "co2_flatline_24h_1": [0.0] * 40,
            "co2_flatline_72h_1": [0.0] * 40,
            "predicted_delta_24h_kg": np.linspace(2.0, 0.1, 40),
            "predicted_delta_48h_kg": np.linspace(3.0, 0.2, 40),
            "predicted_delta_72h_kg": np.linspace(4.0, 0.3, 40),
        }
    )
    for horizon in (24, 48, 72):
        frame[f"predicted_rate_{horizon}h_kg_per_hour"] = (
            frame[f"predicted_delta_{horizon}h_kg"] / horizon
        )

    result, parameters, thresholds = build_provisional_readiness_scores(
        frame,
        horizons_hours=[24, 48, 72],
        component_weights={
            "recent_accumulation": 0.15,
            "weight_position": 0.20,
            "forecast_plateau": 0.25,
            "forecast_slowdown": 0.20,
            "forecast_agreement": 0.10,
            "environmental_stability": 0.10,
        },
        lower_quantile=0.10,
        upper_quantile=0.90,
        plateau_rate_quantile=0.25,
        stability_window_hours=24,
        stability_minimum_periods=6,
        rate_of_change_hours=6,
        class_quantiles={
            "approaching": 0.50,
            "ready": 0.75,
            "high_priority": 0.90,
        },
    )

    assert result["provisional_readiness_score"].between(0.0, 100.0).all()
    assert result["hrsi"].between(0.0, 100.0).all()
    assert np.isfinite(result["provisional_readiness_score"]).all()
    assert parameters["normalization_source_split"] == "train"
    assert thresholds["approaching"] <= thresholds["ready"] <= thresholds["high_priority"]
