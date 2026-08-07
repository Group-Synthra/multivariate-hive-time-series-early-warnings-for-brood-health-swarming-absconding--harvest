import numpy as np
import pandas as pd

from multivari.modules.harvesting.reviewed_dataset import (
    add_post_event_recovery_gap,
    make_future_reviewed_event_target,
)


def test_future_reviewed_target_excludes_current_event() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=200,
        freq="h",
    )
    rows = pd.DataFrame(
        {
            "hive_id": ["h1"] * 200,
            "timestamp": timestamps,
        }
    )
    events = pd.DataFrame(
        {
            "hive_id": ["h1"],
            "event_start": [timestamps[80]],
        }
    )

    result = make_future_reviewed_event_target(
        rows,
        events,
        horizon_hours=72,
        output_column="target",
    )

    assert result.loc[80, "target"] == 0
    assert result.loc[79, "target"] == 1
    assert result.loc[8, "target"] == 1
    assert result.loc[7, "target"] == 0


def test_future_reviewed_target_never_crosses_hives() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=100,
        freq="h",
    )
    rows = pd.DataFrame(
        {
            "hive_id": (
                ["h1"] * 100
                + ["h2"] * 100
            ),
            "timestamp": list(timestamps) * 2,
        }
    )
    events = pd.DataFrame(
        {
            "hive_id": ["h2"],
            "event_start": [timestamps[80]],
        }
    )

    result = make_future_reviewed_event_target(
        rows,
        events,
        horizon_hours=72,
        output_column="target",
    )

    h1 = result.loc[result["hive_id"].eq("h1")]
    assert h1["target"].dropna().sum() == 0


def test_post_event_recovery_gap_includes_event_row() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=50,
        freq="h",
    )
    rows = pd.DataFrame(
        {
            "hive_id": ["h1"] * 50,
            "timestamp": timestamps,
        }
    )
    events = pd.DataFrame(
        {
            "hive_id": ["h1"],
            "event_start": [timestamps[20]],
        }
    )

    result = add_post_event_recovery_gap(
        rows,
        events,
        recovery_hours=24,
    )

    assert result.loc[20, "is_post_event_recovery_gap"]
    assert result.loc[44, "is_post_event_recovery_gap"]
    assert not result.loc[
        45,
        "is_post_event_recovery_gap",
    ]


def test_incomplete_future_horizon_is_missing() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=10,
        freq="h",
    )
    rows = pd.DataFrame(
        {
            "hive_id": ["h1"] * 10,
            "timestamp": timestamps,
        }
    )
    events = pd.DataFrame(
        {
            "hive_id": ["h1"],
            "event_start": [timestamps[8]],
        }
    )

    result = make_future_reviewed_event_target(
        rows,
        events,
        horizon_hours=4,
        output_column="target",
    )

    assert np.isnan(result.loc[6, "target"])
