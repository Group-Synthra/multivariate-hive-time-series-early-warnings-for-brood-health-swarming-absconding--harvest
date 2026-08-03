from __future__ import annotations

import pandas as pd

from .schema import HIVE_COLUMN, SENSOR_COLUMNS, TARGET_COLUMNS, TIMESTAMP_COLUMN


def clean_common_dataset(
    df: pd.DataFrame,
    *,
    interpolate_missing_sensors: bool = False,
    interpolation_limit_rows: int = 3,
) -> pd.DataFrame:
    """Apply only shared, target-agnostic cleaning steps."""
    clean = df.copy()
    clean.columns = [str(column).strip().lower() for column in clean.columns]
    clean[TIMESTAMP_COLUMN] = pd.to_datetime(clean[TIMESTAMP_COLUMN], errors="raise")
    clean[HIVE_COLUMN] = clean[HIVE_COLUMN].astype("string").str.strip()

    for column in SENSOR_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce").astype("float32")
    for column in TARGET_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="raise").astype("int8")

    clean = clean.drop_duplicates(subset=[HIVE_COLUMN, TIMESTAMP_COLUMN], keep="last")
    clean = clean.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)

    if interpolate_missing_sensors:
        clean = _interpolate_sensors(clean, limit=interpolation_limit_rows)

    clean["row_has_missing_sensor"] = clean[list(SENSOR_COLUMNS)].isna().any(axis=1)
    return clean


def _interpolate_sensors(df: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(HIVE_COLUMN, sort=False):
        group = group.sort_values(TIMESTAMP_COLUMN).copy()
        group = group.set_index(TIMESTAMP_COLUMN)
        group[list(SENSOR_COLUMNS)] = group[list(SENSOR_COLUMNS)].interpolate(
            method="time",
            limit=limit,
            limit_area="inside",
        )
        pieces.append(group.reset_index())
    return pd.concat(pieces, ignore_index=True).sort_values(
        [HIVE_COLUMN, TIMESTAMP_COLUMN]
    ).reset_index(drop=True)
