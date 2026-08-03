import pandas as pd

from multivari.common.splitting import assign_chronological_splits


def test_split_is_chronological_within_hive() -> None:
    frame = pd.DataFrame(
        {
            "hive_id": ["h1"] * 100,
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        }
    )
    manifest = assign_chronological_splits(frame, boundary_gap_hours=2)
    assert manifest.iloc[0]["split"] == "train"
    assert manifest.iloc[-1]["split"] == "test"
    assert manifest["is_boundary_gap"].any()
