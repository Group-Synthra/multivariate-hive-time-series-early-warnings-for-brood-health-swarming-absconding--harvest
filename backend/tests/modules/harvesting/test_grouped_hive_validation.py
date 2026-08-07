import pandas as pd

from multivari.modules.harvesting.grouped_hive_validation import (
    create_grouped_positive_hive_folds,
)


def test_grouped_folds_hold_out_one_positive_training_hive() -> None:
    features = pd.DataFrame(
        {
            "hive_id": (
                ["h1"] * 10
                + ["h2"] * 10
                + ["h3"] * 10
                + ["h4"] * 10
                + ["h5"] * 10
                + ["v1"] * 10
            ),
            "split": (
                ["train"] * 50
                + ["validation"] * 10
            ),
            "target": (
                [0] * 9 + [1]
            )
            * 6,
        }
    )
    events = pd.DataFrame(
        {
            "hive_id": ["h1", "h2", "h3", "h4", "h5", "v1"],
            "split": [
                "train",
                "train",
                "train",
                "train",
                "train",
                "validation",
            ],
            "harvest_event_id": [
                "e1",
                "e2",
                "e3",
                "e4",
                "e5",
                "e6",
            ],
        }
    )

    folds, summary = create_grouped_positive_hive_folds(
        features,
        events,
        target_column="target",
        minimum_training_positive_hives=4,
    )

    assert len(folds) == 5
    assert summary["fold_count"] == 5
    assert set(folds["validation_hive_id"]) == {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
    }
    assert folds["training_positive_hive_count"].eq(4).all()
    assert folds["validation_event_count"].eq(1).all()
