from __future__ import annotations

import pandas as pd

from multivari.common.schema import HIVE_COLUMN, TIMESTAMP_COLUMN


def add_harvest_event_identifiers(
    df: pd.DataFrame,
    *,
    source_column: str = "honey_harvested_1",
    event_start_column: str = "harvest_event_start_1",
    event_id_column: str = "harvest_event_id",
) -> pd.DataFrame:
    """Collapse consecutive positive harvest-marker rows into event identifiers."""
    required = {HIVE_COLUMN, TIMESTAMP_COLUMN, source_column}
    missing = sorted(required.difference(df.columns))

    if missing:
        raise ValueError(
            f"Missing columns required for harvest events: {missing}"
        )

    result = df.sort_values(
        [HIVE_COLUMN, TIMESTAMP_COLUMN]
    ).copy()

    observed = set(
        pd.to_numeric(
            result[source_column],
            errors="raise",
        ).unique()
    )

    unexpected = sorted(observed.difference({0, 1}))

    if unexpected:
        raise ValueError(
            f"{source_column} must contain only 0 and 1; "
            f"found {unexpected}"
        )

    result[source_column] = result[source_column].astype("int8")

    previous_marker = (
        result.groupby(
            HIVE_COLUMN,
            sort=False,
        )[source_column]
        .shift(fill_value=0)
    )

    result[event_start_column] = (
        (result[source_column] == 1)
        & (previous_marker == 0)
    ).astype("int8")

    event_number = (
        result.groupby(
            HIVE_COLUMN,
            sort=False,
        )[event_start_column]
        .cumsum()
    )

    positive_mask = result[source_column].eq(1)

    result[event_id_column] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    result.loc[positive_mask, event_id_column] = (
        result.loc[
            positive_mask,
            HIVE_COLUMN,
        ].astype("string")
        + "_harvest_"
        + event_number.loc[
            positive_mask
        ]
        .astype("int32")
        .astype("string")
        .str.zfill(3)
    )

    return result


def build_harvest_event_table(
    df: pd.DataFrame,
    *,
    source_column: str = "honey_harvested_1",
    event_start_column: str = "harvest_event_start_1",
    event_id_column: str = "harvest_event_id",
) -> pd.DataFrame:
    """Return one audit row for every consolidated harvest event."""
    required = {
        HIVE_COLUMN,
        TIMESTAMP_COLUMN,
        source_column,
        event_start_column,
        event_id_column,
    }

    missing = sorted(required.difference(df.columns))

    if missing:
        raise ValueError(
            f"Missing columns required for event table: {missing}"
        )

    event_rows = df.loc[
        df[source_column].eq(1)
        & df[event_id_column].notna()
    ].copy()

    if event_rows.empty:
        return pd.DataFrame(
            columns=[
                HIVE_COLUMN,
                event_id_column,
                "event_start",
                "event_end",
                "positive_rows",
                "event_duration_hours",
            ]
        )

    events = (
        event_rows.groupby(
            [HIVE_COLUMN, event_id_column],
            as_index=False,
        )
        .agg(
            event_start=(TIMESTAMP_COLUMN, "min"),
            event_end=(TIMESTAMP_COLUMN, "max"),
            positive_rows=(source_column, "size"),
        )
        .sort_values(
            [HIVE_COLUMN, "event_start"]
        )
        .reset_index(drop=True)
    )

    events["event_duration_hours"] = (
        (
            events["event_end"]
            - events["event_start"]
        ).dt.total_seconds()
        / 3600
        + 1
    ).astype("float32")

    optional_columns = [
        column
        for column in (
            "split",
            "is_boundary_gap",
        )
        if column in df
    ]

    if optional_columns:
        start_rows = df.loc[
            df[event_start_column].eq(1),
            [
                HIVE_COLUMN,
                event_id_column,
                *optional_columns,
            ],
        ].drop_duplicates(
            [HIVE_COLUMN, event_id_column]
        )

        events = events.merge(
            start_rows,
            on=[
                HIVE_COLUMN,
                event_id_column,
            ],
            how="left",
            validate="one_to_one",
        )

    return events