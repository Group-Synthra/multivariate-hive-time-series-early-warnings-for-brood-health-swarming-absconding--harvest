import numpy as np
import pandas as pd

from multivari.modules.brood_health.features import (
    build_supervised_dataset,
    target_columns,
)
from multivari.modules.brood_health.scoring import (
    BroodHealthScoreConfig,
    compute_score_components,
)


def _frame(hours: int = 110) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=hours, freq="h")
    return pd.DataFrame(
        {
            "hive_id": ["hive-a"] * hours,
            "timestamp": timestamps,
            "temperature_c": 34.5 + np.sin(np.arange(hours) / 8),
            "humidity_pct": 64 + 2 * np.cos(np.arange(hours) / 10),
            "co2_ppm": 900 + 100 * np.sin(np.arange(hours) / 7),
            "weight_kg": 40 + np.arange(hours) * 0.01,
            "brood_health_healthy_1": [1] * hours,
        }
    )


def test_exact_six_hour_target_matches_group_shift() -> None:
    frame = _frame()
    config = BroodHealthScoreConfig()
    scored = compute_score_components(frame, config=config)
    x, y, metadata, feature_columns = build_supervised_dataset(
        frame,
        horizon_hours=6,
        score_config=config,
    )

    expected = scored["brood_health_score"].shift(-6)
    source_timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    source_rows = metadata["timestamp"].map(
        dict(zip(source_timestamps, expected, strict=True))
    )
    np.testing.assert_allclose(
        y["score_t_plus_6h"].to_numpy(),
        source_rows.to_numpy(),
        rtol=0,
        atol=1e-10,
    )
    assert target_columns(6)[-1] == "score_t_plus_6h"
    assert len(x) == len(y) == len(metadata)
    assert feature_columns


def test_safety_minimum_is_secondary_trajectory_minimum() -> None:
    frame = _frame()
    _, y, metadata, _ = build_supervised_dataset(frame, horizon_hours=6)
    np.testing.assert_allclose(
        metadata["minimum_future_score"].to_numpy(),
        y.min(axis=1).to_numpy(),
    )
    np.testing.assert_allclose(
        metadata["exact_future_score"].to_numpy(),
        y["score_t_plus_6h"].to_numpy(),
    )
