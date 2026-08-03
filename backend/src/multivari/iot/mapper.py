from __future__ import annotations

from collections.abc import Mapping
import pandas as pd

from multivari.common.schema import HIVE_COLUMN, SENSOR_COLUMNS, TIMESTAMP_COLUMN


def map_iot_to_canonical(
    raw: pd.DataFrame,
    *,
    column_mapping: Mapping[str, str],
) -> pd.DataFrame:
    """Rename live PostgreSQL columns to the shared historical-data contract."""
    missing_source = [column for column in column_mapping if column not in raw.columns]
    if missing_source:
        raise ValueError(f"Live IoT data is missing source columns: {missing_source}")

    mapped = raw.rename(columns=dict(column_mapping)).copy()
    mapped[TIMESTAMP_COLUMN] = pd.to_datetime(mapped[TIMESTAMP_COLUMN], utc=True)
    mapped[HIVE_COLUMN] = mapped[HIVE_COLUMN].astype("string").str.strip()

    for column in SENSOR_COLUMNS:
        mapped[column] = pd.to_numeric(mapped[column], errors="coerce")

    return mapped.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)


def aggregate_live_to_training_frequency(
    live: pd.DataFrame,
    *,
    frequency: str = "1h",
) -> pd.DataFrame:
    """Aggregate 10-minute live readings to the hourly frequency used for training."""
    indexed = live.set_index(TIMESTAMP_COLUMN)
    aggregated = (
        indexed.groupby(HIVE_COLUMN)[list(SENSOR_COLUMNS)]
        .resample(frequency)
        .median()
        .reset_index()
    )
    return aggregated.dropna(subset=list(SENSOR_COLUMNS), how="all")
