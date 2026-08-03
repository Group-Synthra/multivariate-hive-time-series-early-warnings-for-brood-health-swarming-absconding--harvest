import pandas as pd

from multivari.common.targets import make_future_event_target


def test_future_event_target() -> None:
    frame = pd.DataFrame(
        {
            "hive_id": ["h1"] * 6,
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
            "event": [0, 0, 0, 1, 0, 0],
        }
    )
    result = make_future_event_target(
        frame,
        event_column="event",
        horizon_hours=2,
        output_column="future_event",
    )
    assert result.loc[1, "future_event"] == 1
