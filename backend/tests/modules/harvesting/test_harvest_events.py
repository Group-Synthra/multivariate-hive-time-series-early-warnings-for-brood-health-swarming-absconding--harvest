import pandas as pd

from multivari.modules.harvesting.events import (
    add_harvest_event_identifiers,
    build_harvest_event_table,
)


def test_consecutive_positive_rows_form_one_event() -> None:
    frame = pd.DataFrame(
        {
            "hive_id": ["h1"] * 6,
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=6,
                freq="h",
            ),
            "honey_harvested_1": [
                0,
                1,
                1,
                0,
                1,
                0,
            ],
        }
    )

    result = add_harvest_event_identifiers(frame)

    assert result["harvest_event_start_1"].tolist() == [
        0,
        1,
        0,
        0,
        1,
        0,
    ]

    assert (
        result.loc[
            1,
            "harvest_event_id",
        ]
        == "h1_harvest_001"
    )

    assert (
        result.loc[
            2,
            "harvest_event_id",
        ]
        == "h1_harvest_001"
    )

    assert (
        result.loc[
            4,
            "harvest_event_id",
        ]
        == "h1_harvest_002"
    )

    events = build_harvest_event_table(result)

    assert len(events) == 2
    assert events.loc[0, "positive_rows"] == 2


def test_event_numbering_is_separate_for_each_hive() -> None:
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
            "honey_harvested_1": [
                0,
                1,
                0,
                1,
            ],
        }
    )

    result = add_harvest_event_identifiers(frame)

    observed = result.loc[
        result["honey_harvested_1"].eq(1),
        "harvest_event_id",
    ].tolist()

    assert observed == [
        "h1_harvest_001",
        "h2_harvest_001",
    ]
