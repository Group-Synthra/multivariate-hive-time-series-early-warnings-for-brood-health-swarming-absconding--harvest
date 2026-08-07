import numpy as np
import pandas as pd

from multivari.modules.harvesting.alert_policy_refinement import (
    add_contiguous_segment_id,
    apply_alert_policy,
    build_event_detection_table,
    select_alert_policy,
)


def test_segments_restart_after_gap_and_between_hives() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 01:00",
                    "2024-01-01 05:00",
                    "2024-01-01 00:00",
                ]
            ),
            "hive_id": ["h1", "h1", "h1", "h2"],
            "raw_probability": [0.1, 0.2, 0.3, 0.4],
        }
    )

    result = add_contiguous_segment_id(frame)

    h1_segments = result.loc[
        result["hive_id"].eq("h1"),
        "_segment_id",
    ].tolist()
    h2_segment = result.loc[
        result["hive_id"].eq("h2"),
        "_segment_id",
    ].iloc[0]

    assert h1_segments[0] == h1_segments[1]
    assert h1_segments[1] != h1_segments[2]
    assert h2_segment != h1_segments[0]


def test_alert_requires_consecutive_threshold_hours() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=6,
                freq="h",
            ),
            "hive_id": ["h1"] * 6,
            "raw_probability": [0.9, 0.2, 0.9, 0.9, 0.9, 0.2],
            "target": [0] * 6,
            "split": ["validation"] * 6,
        }
    )

    result = apply_alert_policy(
        frame,
        probability_column="raw_probability",
        smoothing_window_hours=1,
        threshold=0.8,
        minimum_consecutive_hours=3,
    )

    assert result["alert"].tolist() == [0, 0, 0, 0, 1, 0]


def test_event_detection_excludes_event_timestamp() -> None:
    event_time = pd.Timestamp("2024-01-04 00:00")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                event_time - pd.Timedelta(hours=72),
                event_time,
                freq="h",
            ),
            "hive_id": ["h1"] * 73,
            "raw_probability": [0.1] * 71 + [0.9, 0.9],
            "smoothed_probability": [0.1] * 71 + [0.9, 0.9],
            "alert": [0] * 71 + [1, 1],
        }
    )
    events = pd.DataFrame(
        {
            "hive_id": ["h1"],
            "harvest_event_id": ["e1"],
            "harvest_session_id": ["s1"],
            "event_start": [event_time],
            "split": ["validation"],
        }
    )

    result = build_event_detection_table(
        frame,
        events,
        split="validation",
        horizon_hours=72,
    )

    assert result.loc[0, "detected"]
    assert result.loc[0, "lead_hours"] == 1.0
    assert result.loc[0, "alert_rows"] == 1


def test_policy_selection_prefers_fewer_false_alerts() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=160,
        freq="h",
    )
    event_time = timestamps[120]
    probabilities = np.full(160, 0.05)
    probabilities[20:23] = 0.7
    probabilities[80:82] = 0.7
    probabilities[112:120] = 0.9

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * 160,
            "split": ["validation"] * 160,
            "target": [
                int(
                    timestamp < event_time
                    and timestamp
                    >= event_time - pd.Timedelta(hours=72)
                )
                for timestamp in timestamps
            ],
            "raw_probability": probabilities,
        }
    )
    events = pd.DataFrame(
        {
            "hive_id": ["h1"],
            "harvest_event_id": ["e1"],
            "harvest_session_id": ["s1"],
            "event_start": [event_time],
            "split": ["validation"],
        }
    )

    policy, sweep, selected, detection = select_alert_policy(
        frame,
        events,
        target_column="target",
        probability_column="raw_probability",
        smoothing_windows_hours=[1],
        minimum_consecutive_hours=[1, 3],
        threshold_grid_points=21,
        minimum_validation_event_recall=1.0,
        minimum_median_lead_hours=1.0,
        false_alert_gap_hours=6,
        horizon_hours=72,
    )

    assert not sweep.empty
    assert detection["detected"].all()
    assert selected["alert"].sum() > 0
    selected_row = sweep.loc[
        sweep["smoothing_window_hours"].eq(
            policy["smoothing_window_hours"]
        )
        & sweep["minimum_consecutive_hours"].eq(
            policy["minimum_consecutive_hours"]
        )
        & np.isclose(
            sweep["threshold"],
            policy["threshold"],
        )
    ].iloc[0]
    eligible = sweep.loc[
        sweep["event_recall"].eq(1.0)
        & sweep["median_lead_hours"].ge(1.0)
    ]
    assert selected_row["false_alert_episodes"] == (
        eligible["false_alert_episodes"].min()
    )


def test_smoothing_does_not_cross_gap() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 01:00",
                    "2024-01-01 02:00",
                    "2024-01-01 08:00",
                    "2024-01-01 09:00",
                    "2024-01-01 10:00",
                ]
            ),
            "hive_id": ["h1"] * 6,
            "split": ["validation"] * 6,
            "target": [0] * 6,
            "raw_probability": [0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
        }
    )

    result = apply_alert_policy(
        frame,
        probability_column="raw_probability",
        smoothing_window_hours=3,
        threshold=0.5,
        minimum_consecutive_hours=1,
    )

    second_segment = result.loc[
        result["timestamp"].ge(
            pd.Timestamp("2024-01-01 08:00")
        )
    ]

    assert pd.isna(second_segment.iloc[0]["smoothed_probability"])
    assert pd.isna(second_segment.iloc[1]["smoothed_probability"])
    assert second_segment.iloc[2]["smoothed_probability"] == 0.1
