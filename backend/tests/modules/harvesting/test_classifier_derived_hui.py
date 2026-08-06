from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multivari.modules.harvesting.classifier_derived_hui import (
    add_future_hui_target,
    assign_harvest_readiness_class,
    evaluate_future_hui_research_gate,
    probability_to_hui,
    regression_metrics,
)

PROBABILITY_ANCHORS = [
    0.0,
    0.0008187403977683789,
    0.0034394735186074887,
    0.008824379298130958,
    0.01943345880678582,
    0.05090735328716948,
    0.3637653093683719,
]
HUI_ANCHORS = [0.0, 20.0, 40.0, 60.0, 70.0, 80.0, 100.0]
CLASS_CONFIG = {
    "not_ready_upper": 40.0,
    "approaching_upper": 60.0,
    "ready_upper": 80.0,
}


def test_probability_to_hui_preserves_frozen_anchors() -> None:
    transformed = probability_to_hui(
        np.asarray(PROBABILITY_ANCHORS),
        probability_anchors=PROBABILITY_ANCHORS,
        hui_anchors=HUI_ANCHORS,
    )
    np.testing.assert_allclose(transformed, HUI_ANCHORS)


def test_probability_to_hui_is_monotonic_and_bounded() -> None:
    scores = np.linspace(-0.1, 1.1, 1000)
    hui = probability_to_hui(
        scores,
        probability_anchors=PROBABILITY_ANCHORS,
        hui_anchors=HUI_ANCHORS,
    )
    assert hui.min() == pytest.approx(0.0)
    assert hui.max() == pytest.approx(100.0)
    assert np.all(np.diff(hui) >= 0.0)


def test_readiness_classes_follow_fixed_boundaries() -> None:
    values = pd.Series([0.0, 39.999, 40.0, 59.999, 60.0, 79.999, 80.0, 100.0])
    classes = assign_harvest_readiness_class(
        values,
        **CLASS_CONFIG,
    ).tolist()
    assert classes == [
        "Not Ready",
        "Not Ready",
        "Approaching Harvest",
        "Approaching Harvest",
        "Ready",
        "Ready",
        "High-Priority Harvest",
        "High-Priority Harvest",
    ]


def test_future_target_does_not_cross_gap_or_split() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 02:00",
                    "2026-01-01 04:00",
                    "2026-01-01 05:00",
                    "2026-01-01 06:00",
                ]
            ),
            "hive_id": ["h1"] * 6,
            "split": ["train", "train", "validation", "validation", "validation", "validation"],
            "classifier_derived_hui": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    result = add_future_hui_target(frame, horizon_hours=1)
    target = "future_classifier_derived_hui_1h"

    assert result.loc[0, target] == pytest.approx(20.0)
    assert pd.isna(result.loc[1, target])
    assert pd.isna(result.loc[2, target])
    assert result.loc[3, target] == pytest.approx(50.0)
    assert result.loc[4, target] == pytest.approx(60.0)


def test_regression_metrics_include_class_agreement() -> None:
    metrics = regression_metrics(
        np.array([20.0, 50.0, 70.0, 90.0]),
        np.array([22.0, 58.0, 65.0, 81.0]),
        class_config=CLASS_CONFIG,
    )
    assert metrics["mae"] == pytest.approx(6.0)
    assert metrics["within_10_points_fraction"] == pytest.approx(1.0)
    assert metrics["readiness_class_agreement_fraction"] == pytest.approx(1.0)


def test_future_hui_gate_requires_multiple_improved_horizons() -> None:
    comparison = pd.DataFrame(
        [
            {"horizon_hours": 24, "model": "persistence", "status": "ok", "validation_mae": 10.0},
            {"horizon_hours": 24, "model": "ridge", "status": "ok", "validation_mae": 8.0},
            {"horizon_hours": 48, "model": "persistence", "status": "ok", "validation_mae": 10.0},
            {"horizon_hours": 48, "model": "ridge", "status": "ok", "validation_mae": 9.8},
            {"horizon_hours": 72, "model": "persistence", "status": "ok", "validation_mae": 10.0},
            {"horizon_hours": 72, "model": "ridge", "status": "ok", "validation_mae": 8.5},
        ]
    )
    summary = {
        "horizons": {
            "24": {
                "selected_model": "ridge",
                "selected_feature_set": "hui_history_only",
                "validation": {"mae": 8.0},
                "test": {"mae": 9.0},
            },
            "48": {
                "selected_model": "ridge",
                "selected_feature_set": "hui_history_only",
                "validation": {"mae": 9.8},
                "test": {"mae": 10.0},
            },
            "72": {
                "selected_model": "ridge",
                "selected_feature_set": "hui_history_only",
                "validation": {"mae": 8.5},
                "test": {"mae": 9.5},
            },
        }
    }
    gate = evaluate_future_hui_research_gate(
        comparison,
        summary,
        horizons_hours=[24, 48, 72],
        minimum_improvement=0.05,
        required_improved_horizons=2,
        maximum_test_to_validation_ratio=2.0,
    )
    assert gate["gate_passed"] is True
    assert gate["improved_horizon_count"] == 2
