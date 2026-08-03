from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import HIVE_COLUMN, TIMESTAMP_COLUMN


def make_future_point_target(
    df: pd.DataFrame,
    *,
    source_column: str,
    horizon_hours: int,
    output_column: str,
) -> pd.DataFrame:
    """Create a target equal to a value observed a fixed number of hourly rows ahead."""
    result = df.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN]).copy()
    result[output_column] = result.groupby(HIVE_COLUMN, sort=False)[source_column].shift(
        -horizon_hours
    )
    return result


def make_future_event_target(
    df: pd.DataFrame,
    *,
    event_column: str,
    horizon_hours: int,
    output_column: str,
) -> pd.DataFrame:
    """Create 'event occurs within the next horizon' labels without mixing hives."""
    result = df.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN]).copy()
    target = pd.Series(np.nan, index=result.index, dtype="float32")

    for _, group in result.groupby(HIVE_COLUMN, sort=False):
        values = group[event_column].to_numpy(dtype="float32")
        out = np.full(len(values), np.nan, dtype="float32")
        for position in range(max(0, len(values) - horizon_hours)):
            out[position] = values[position + 1 : position + horizon_hours + 1].max()
        target.loc[group.index] = out

    result[output_column] = target
    return result


def remove_target_leakage_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows reserved around split boundaries before module training."""
    if "is_boundary_gap" not in df.columns:
        raise ValueError("Join the common split manifest before removing boundary rows.")
    return df.loc[~df["is_boundary_gap"]].copy()
