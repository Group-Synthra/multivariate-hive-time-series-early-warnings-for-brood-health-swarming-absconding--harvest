from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import HIVE_COLUMN, TIMESTAMP_COLUMN


def assign_chronological_splits(
    df: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    boundary_gap_hours: int = 72,
) -> pd.DataFrame:
    """Assign train/validation/test within each hive without random row shuffling."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")

    manifest_parts: list[pd.DataFrame] = []
    for hive_id, group in df.groupby(HIVE_COLUMN, sort=False):
        ordered = group[[HIVE_COLUMN, TIMESTAMP_COLUMN]].sort_values(TIMESTAMP_COLUMN).copy()
        size = len(ordered)
        train_end = int(np.floor(size * train_fraction))
        validation_end = int(np.floor(size * (train_fraction + validation_fraction)))

        ordered["split"] = "test"
        ordered.iloc[:train_end, ordered.columns.get_loc("split")] = "train"
        ordered.iloc[train_end:validation_end, ordered.columns.get_loc("split")] = "validation"

        # Reserve a gap around split boundaries for future-label horizons and rolling histories.
        ordered["is_boundary_gap"] = False
        for boundary in (train_end, validation_end):
            lower = max(0, boundary - boundary_gap_hours)
            upper = min(size, boundary + boundary_gap_hours)
            ordered.iloc[lower:upper, ordered.columns.get_loc("is_boundary_gap")] = True

        ordered[HIVE_COLUMN] = hive_id
        manifest_parts.append(ordered)

    return pd.concat(manifest_parts, ignore_index=True)


def join_split_manifest(df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        manifest,
        on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        how="left",
        validate="one_to_one",
    )
