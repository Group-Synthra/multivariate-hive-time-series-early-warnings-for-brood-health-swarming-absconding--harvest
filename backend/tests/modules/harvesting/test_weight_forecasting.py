import numpy as np
import pandas as pd

from multivari.modules.harvesting.weight_forecasting import (
    PersistenceDeltaRegressor,
    RecentTrendDeltaRegressor,
    build_feature_sets,
    build_future_weight_targets,
    calculate_regression_metrics,
)


def test_future_target_uses_exact_same_hive_timestamp() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=30,
        freq="h",
    )
    clean = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * 30,
            "weight_kg": np.arange(30, dtype=float),
        }
    )
    split = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * 30,
            "split": ["train"] * 30,
        }
    )
    features = pd.DataFrame(
        {
            "timestamp": timestamps[:5],
            "hive_id": ["h1"] * 5,
            "split": ["train"] * 5,
            "weight_kg_current": np.arange(5, dtype=float),
        }
    )

    result = build_future_weight_targets(
        features,
        clean,
        split,
        horizons_hours=[24],
    )

    assert (
        result["weight_delta_next_24h_kg"]
        .dropna()
        .eq(24.0)
        .all()
    )


def test_future_target_does_not_cross_split() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=30,
        freq="h",
    )
    clean = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * 30,
            "weight_kg": np.arange(30, dtype=float),
        }
    )
    split = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * 30,
            "split": ["train"] * 24 + ["validation"] * 6,
        }
    )
    features = pd.DataFrame(
        {
            "timestamp": [timestamps[0]],
            "hive_id": ["h1"],
            "split": ["train"],
            "weight_kg_current": [0.0],
        }
    )

    result = build_future_weight_targets(
        features,
        clean,
        split,
        horizons_hours=[24],
    )

    assert pd.isna(
        result.loc[0, "weight_delta_next_24h_kg"]
    )


def test_feature_ablation_excludes_humidity() -> None:
    features = [
        "weight_delta_24h_kg",
        "humidity_pct_mean_24h",
        "co2_ppm_mean_24h",
    ]
    result = build_feature_sets(
        features,
        {
            "no_humidity": {
                "include_all": True,
                "exclude_prefixes": ["humidity_"],
            }
        },
    )

    assert result["no_humidity"] == [
        "weight_delta_24h_kg",
        "co2_ppm_mean_24h",
    ]


def test_persistence_predicts_zero_delta() -> None:
    features = pd.DataFrame(
        {"weight_kg_current": [10.0, 11.0]}
    )
    estimator = PersistenceDeltaRegressor().fit(
        features,
        pd.Series([1.0, 2.0]),
    )

    assert np.array_equal(
        estimator.predict(features),
        np.array([0.0, 0.0]),
    )


def test_recent_trend_extends_past_only_slope() -> None:
    features = pd.DataFrame(
        {
            "weight_trend_24h_kg_per_hour": [
                0.1,
                -0.05,
            ]
        }
    )
    estimator = RecentTrendDeltaRegressor(
        trend_feature=(
            "weight_trend_24h_kg_per_hour"
        ),
        horizon_hours=24,
    ).fit(features, pd.Series([0.0, 0.0]))

    prediction = estimator.predict(features)

    assert np.allclose(prediction, [2.4, -1.2])


def test_regression_metrics_report_bias_direction() -> None:
    metrics = calculate_regression_metrics(
        np.array([1.0, 2.0]),
        np.array([2.0, 3.0]),
    )

    assert metrics["mae"] == 1.0
    assert metrics["bias"] == 1.0
