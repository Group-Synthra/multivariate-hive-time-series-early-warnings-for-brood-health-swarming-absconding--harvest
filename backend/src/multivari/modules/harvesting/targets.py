from __future__ import annotations

import pandas as pd

from multivari.common.targets import make_future_event_target


def add_future_harvest_target(
    df: pd.DataFrame,
    *,
    event_start_column: str = "harvest_event_start_1",
    horizon_hours: int = 72,
    output_column: str = "harvest_within_next_72h",
) -> pd.DataFrame:
    """Create a future harvest target that excludes the current timestamp."""
    if horizon_hours <= 0:
        raise ValueError(
            "horizon_hours must be greater than zero"
        )

    if event_start_column not in df.columns:
        raise ValueError(
            f"Missing event-start column: {event_start_column}"
        )

    return make_future_event_target(
        df,
        event_column=event_start_column,
        horizon_hours=horizon_hours,
        output_column=output_column,
    )