import numpy as np
import pandas as pd

from multivari.modules.harvesting.research_model_comparison import (
    attach_future_event_metadata,
    build_feature_sets,
    calculate_session_balanced_weights,
    cluster_harvest_sessions,
    select_operating_threshold,
)


def test_close_events_are_grouped_into_one_session() -> None:
    events = pd.DataFrame(
        {
            "hive_id": ["h1", "h2", "h3"],
            "harvest_event_id": ["e1", "e2", "e3"],
            "event_start": pd.to_datetime(
                [
                    "2024-01-01 10:00",
                    "2024-01-01 14:00",
                    "2024-01-03 10:00",
                ]
            ),
            "split": ["train", "train", "train"],
        }
    )

    result = cluster_harvest_sessions(
        events,
        session_gap_hours=12,
    )

    assert result.loc[0, "harvest_session_id"] == (
        result.loc[1, "harvest_session_id"]
    )
    assert result.loc[1, "harvest_session_id"] != (
        result.loc[2, "harvest_session_id"]
    )


def test_positive_rows_attach_to_future_event_only() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=100,
        freq="h",
    )
    rows = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * 100,
            "target": [0] * 8 + [1] * 72 + [0] * 20,
        }
    )
    events = pd.DataFrame(
        {
            "hive_id": ["h1"],
            "harvest_event_id": ["e1"],
            "harvest_session_id": ["s1"],
            "event_start": [timestamps[80]],
        }
    )

    result = attach_future_event_metadata(
        rows,
        events,
        target_column="target",
        horizon_hours=72,
    )

    assert pd.isna(result.loc[80, "harvest_event_id"])
    assert result.loc[79, "harvest_event_id"] == "e1"
    assert result.loc[8, "harvest_event_id"] == "e1"


def test_session_balanced_weights_equalize_sessions() -> None:
    rows = pd.DataFrame(
        {
            "target": [1, 1, 1, 1, 1, 1, 0, 0],
            "harvest_event_id": [
                "e1",
                "e1",
                "e2",
                "e2",
                "e2",
                "e3",
                pd.NA,
                pd.NA,
            ],
            "harvest_session_id": [
                "s1",
                "s1",
                "s1",
                "s1",
                "s1",
                "s2",
                pd.NA,
                pd.NA,
            ],
        }
    )

    weights = calculate_session_balanced_weights(
        rows,
        target_column="target",
    )
    rows["weight"] = weights

    positive_by_session = (
        rows.loc[rows["target"].eq(1)]
        .groupby("harvest_session_id")["weight"]
        .sum()
    )

    assert np.isclose(weights.mean(), 1.0)
    assert np.isclose(positive_by_session["s1"], 2.0)
    assert np.isclose(positive_by_session["s2"], 2.0)
    assert np.isclose(
        rows.loc[rows["target"].eq(0), "weight"].sum(),
        4.0,
    )


def test_feature_set_ablation_excludes_humidity() -> None:
    available = [
        "weight_delta_24h_kg",
        "humidity_pct_mean_24h",
        "co2_ppm_mean_24h",
    ]
    config = {
        "no_humidity": {
            "exclude_prefixes": ["humidity_"],
            "include_all": True,
        }
    }

    result = build_feature_sets(available, config)

    assert result["no_humidity"] == [
        "weight_delta_24h_kg",
        "co2_ppm_mean_24h",
    ]


def test_threshold_selection_can_require_event_recall() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=100,
        freq="h",
    )
    predictions = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * 100,
            "split": ["validation"] * 100,
            "target": [0] * 28 + [1] * 72,
            "probability": np.linspace(0.01, 0.99, 100),
        }
    )
    events = pd.DataFrame(
        {
            "hive_id": ["h1"],
            "harvest_event_id": ["e1"],
            "harvest_session_id": ["s1"],
            "event_start": [timestamps[-1] + pd.Timedelta(hours=1)],
            "split": ["validation"],
        }
    )

    threshold, sweep, detection = select_operating_threshold(
        predictions,
        events,
        target_column="target",
        probability_column="probability",
        horizon_hours=72,
        false_alert_gap_hours=6,
        threshold_grid_points=31,
        minimum_event_recall=1.0,
    )

    assert 0 < threshold < 1
    assert not sweep.empty
    assert detection["detected"].all()
