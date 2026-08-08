import pandas as pd
import pytest

from multivari.modules.harvesting.review import (
    build_reviewed_event_table,
    create_manual_review_template,
    validate_manual_review,
)


def test_template_suggests_candidate_for_delayed_event() -> None:
    audit = pd.DataFrame(
        {
            "hive_id": ["h1"],
            "harvest_event_id": ["event_1"],
            "marker_event_start": ["2024-01-03 10:00:00"],
            "candidate_drop_onset": ["2024-01-01 10:00:00"],
            "marker_delay_hours": [48.0],
            "alignment_status": ["marker_delayed"],
            "persistent_drop_kg": [5.0],
            "co2_flatline_pre72h": [0],
        }
    )

    template = create_manual_review_template(audit)

    assert (
        template.loc[
            0,
            "suggested_include_for_training",
        ]
        == 1
    )
    assert (
        template.loc[
            0,
            "suggested_reviewed_event_start",
        ]
        == "2024-01-01 10:00:00"
    )


def test_incomplete_review_is_rejected() -> None:
    review = pd.DataFrame(
        {
            "hive_id": ["h1"],
            "harvest_event_id": ["event_1"],
            "manual_event_type": [""],
            "manual_include_for_training": [""],
            "manual_reviewed_event_start": [""],
            "manual_review_complete": [0],
        }
    )

    with pytest.raises(ValueError):
        validate_manual_review(review)


def test_reviewed_event_must_match_common_timestamp() -> None:
    review = pd.DataFrame(
        {
            "hive_id": ["h1"],
            "harvest_event_id": ["event_1"],
            "manual_event_type": ["probable_harvest"],
            "manual_include_for_training": [1],
            "manual_reviewed_event_start": ["2024-01-01 10:30:00"],
            "manual_review_complete": [1],
            "alignment_status": ["marker_delayed"],
            "marker_event_start": ["2024-01-03 10:00:00"],
            "candidate_drop_onset": ["2024-01-01 10:00:00"],
            "marker_delay_hours": [48.0],
            "persistent_drop_kg": [5.0],
            "co2_flatline_pre72h": [0],
        }
    )
    common = pd.DataFrame(
        {
            "hive_id": ["h1"],
            "timestamp": pd.to_datetime(["2024-01-01 10:00:00"]),
        }
    )
    manifest = common.assign(
        split="train",
        is_boundary_gap=False,
    )

    with pytest.raises(ValueError):
        build_reviewed_event_table(
            review,
            common,
            manifest,
        )
