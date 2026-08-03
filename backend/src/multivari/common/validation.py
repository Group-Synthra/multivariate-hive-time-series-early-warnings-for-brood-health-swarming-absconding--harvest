from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .schema import (
    HIVE_COLUMN,
    REQUIRED_COLUMNS,
    SENSOR_COLUMNS,
    SENSOR_SANITY_BOUNDS,
    TARGET_COLUMNS,
    TIMESTAMP_COLUMN,
)


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    hives: int
    duplicate_hive_timestamps: int
    missing_by_column: dict[str, int]
    invalid_binary_values: dict[str, list[Any]]
    out_of_sanity_bounds: dict[str, int]
    timestamp_start: str | None
    timestamp_end: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def profile_and_validate(df: pd.DataFrame) -> ValidationReport:
    validate_required_columns(df)

    parsed_timestamp = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")
    if parsed_timestamp.isna().any():
        raise ValueError("The timestamp column contains invalid values.")
    if df[HIVE_COLUMN].isna().any():
        raise ValueError("The hive_id column contains missing values.")

    invalid_binary: dict[str, list[Any]] = {}
    for column in TARGET_COLUMNS:
        observed = set(pd.to_numeric(df[column], errors="coerce").dropna().unique().tolist())
        unexpected = sorted(observed.difference({0, 1}))
        if unexpected:
            invalid_binary[column] = unexpected

    bounds_report: dict[str, int] = {}
    for column in SENSOR_COLUMNS:
        numeric = pd.to_numeric(df[column], errors="coerce")
        low, high = SENSOR_SANITY_BOUNDS[column]
        bounds_report[column] = int(((numeric < low) | (numeric > high)).sum())

    return ValidationReport(
        rows=len(df),
        hives=int(df[HIVE_COLUMN].nunique()),
        duplicate_hive_timestamps=int(
            df.duplicated(subset=[HIVE_COLUMN, TIMESTAMP_COLUMN]).sum()
        ),
        missing_by_column={column: int(df[column].isna().sum()) for column in REQUIRED_COLUMNS},
        invalid_binary_values=invalid_binary,
        out_of_sanity_bounds=bounds_report,
        timestamp_start=parsed_timestamp.min().isoformat() if len(df) else None,
        timestamp_end=parsed_timestamp.max().isoformat() if len(df) else None,
    )
