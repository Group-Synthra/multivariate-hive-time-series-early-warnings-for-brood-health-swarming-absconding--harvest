import pandas as pd

from multivari.modules.harvesting.targets import (
    add_future_harvest_target,
)


def test_future_target_excludes_current_event() -> None:
    frame = pd.DataFrame(
        {
            "hive_id": ["h1"] * 6,
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=6,
                freq="h",
            ),
            "harvest_event_start_1": [
                0,
                0,
                0,
                1,
                0,
                0,
            ],
        }
    )

    result = add_future_harvest_target(
        frame,
        horizon_hours=2,
    )

    assert (
        result.loc[
            1,
            "harvest_within_next_72h",
        ]
        == 1
    )

    assert (
        result.loc[
            2,
            "harvest_within_next_72h",
        ]
        == 1
    )

    # The current event is not treated as a future event.
    assert (
        result.loc[
            3,
            "harvest_within_next_72h",
        ]
        == 0
    )

    # There is not enough future history for these rows.
    assert pd.isna(
        result.loc[
            4,
            "harvest_within_next_72h",
        ]
    )

    assert pd.isna(
        result.loc[
            5,
            "harvest_within_next_72h",
        ]
    )


def test_future_target_never_crosses_hive_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "hive_id": [
                "h1",
                "h1",
                "h2",
                "h2",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 01:00",
                ]
                * 2
            ),
            "harvest_event_start_1": [
                0,
                0,
                1,
                0,
            ],
        }
    )

    result = add_future_harvest_target(
        frame,
        horizon_hours=1,
    )

    h1_rows = result.loc[result["hive_id"].eq("h1")]

    assert h1_rows["harvest_within_next_72h"].fillna(0).sum() == 0
