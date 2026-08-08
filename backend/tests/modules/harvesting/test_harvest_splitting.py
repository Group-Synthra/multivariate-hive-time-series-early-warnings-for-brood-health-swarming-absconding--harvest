import pandas as pd

from multivari.modules.harvesting.splitting import (
    create_event_aware_folds,
    folds_to_frame,
)


def test_event_aware_folds_are_chronological() -> None:
    events = pd.DataFrame(
        {
            "split": ["train"] * 30,
            "event_start": pd.date_range(
                "2024-01-01",
                periods=30,
                freq="7D",
            ),
            "is_boundary_gap": [False] * 30,
        }
    )

    folds = create_event_aware_folds(
        events,
        minimum_training_events=12,
        validation_events_per_fold=6,
        prediction_horizon_hours=72,
        purge_hours=72,
    )

    assert len(folds) >= 1

    for fold in folds:
        assert fold.train_end < fold.validation_start
        assert fold.training_events >= 12
        assert fold.validation_events <= 6


def test_only_official_training_events_are_used() -> None:
    events = pd.DataFrame(
        {
            "split": (["train"] * 24 + ["validation"] * 3 + ["test"] * 2),
            "event_start": pd.date_range(
                "2024-01-01",
                periods=29,
                freq="7D",
            ),
            "is_boundary_gap": [False] * 29,
        }
    )

    folds = create_event_aware_folds(
        events,
        minimum_training_events=12,
        validation_events_per_fold=6,
        prediction_horizon_hours=72,
        purge_hours=72,
    )

    frame = folds_to_frame(folds)

    assert not frame.empty
    assert frame["training_events"].max() <= 24
