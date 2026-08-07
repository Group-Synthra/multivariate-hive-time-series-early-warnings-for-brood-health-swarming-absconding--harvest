import numpy as np
import pandas as pd

from multivari.modules.harvesting.robust_weight_forecasting import (
    PersistenceDeltaRegressor,
    build_robust_future_targets,
    calculate_regression_metrics,
)


def test_robust_target_uses_trailing_median_endpoints() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=40,
        freq="h",
    )
    clean = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * 40,
            "weight_kg": np.arange(40, dtype=float),
        }
    )
    split = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * 40,
            "split": ["train"] * 40,
        }
    )
    features = pd.DataFrame(
        {
            "timestamp": timestamps[5:10],
            "hive_id": ["h1"] * 5,
            "split": ["train"] * 5,
        }
    )

    result = build_robust_future_targets(
        features,
        clean,
        split,
        horizons_hours=[24],
        target_window_hours=6,
    )

    assert result["robust_weight_delta_next_24h_kg"].eq(24.0).all()


def test_robust_target_does_not_cross_split() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=40,
        freq="h",
    )
    clean = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * 40,
            "weight_kg": np.arange(40, dtype=float),
        }
    )
    split = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * 40,
            "split": ["train"] * 24 + ["validation"] * 16,
        }
    )
    features = pd.DataFrame(
        {
            "timestamp": [timestamps[5]],
            "hive_id": ["h1"],
            "split": ["train"],
        }
    )

    result = build_robust_future_targets(
        features,
        clean,
        split,
        horizons_hours=[24],
        target_window_hours=6,
    )

    assert pd.isna(
        result.loc[
            0,
            "robust_weight_delta_next_24h_kg",
        ]
    )


def test_robust_target_rejects_gap_in_endpoint_window() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=35,
        freq="h",
    ).delete(28)
    clean = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * len(timestamps),
            "weight_kg": np.arange(len(timestamps), dtype=float),
        }
    )
    split = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * len(timestamps),
            "split": ["train"] * len(timestamps),
        }
    )
    features = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01 05:00")],
            "hive_id": ["h1"],
            "split": ["train"],
        }
    )

    result = build_robust_future_targets(
        features,
        clean,
        split,
        horizons_hours=[24],
        target_window_hours=6,
    )

    assert pd.isna(
        result.loc[
            0,
            "robust_weight_delta_next_24h_kg",
        ]
    )


def test_persistence_predicts_zero_robust_delta() -> None:
    features = pd.DataFrame({"x": [1.0, 2.0]})
    estimator = PersistenceDeltaRegressor().fit(
        features,
        pd.Series([0.5, -0.2]),
    )

    assert np.array_equal(
        estimator.predict(features),
        np.zeros(2),
    )


def test_regression_metrics_report_positive_bias() -> None:
    metrics = calculate_regression_metrics(
        np.array([1.0, 2.0]),
        np.array([2.0, 3.0]),
    )

    assert metrics["mae"] == 1.0
    assert metrics["bias"] == 1.0
