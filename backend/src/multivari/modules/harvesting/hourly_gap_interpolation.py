from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def interpolate_bounded_hourly_gaps(
    hourly: pd.DataFrame,
    *,
    hive_column: str,
    timestamp_column: str,
    sensor_columns: list[str],
    required_sensor_columns: list[str],
    max_gap_hours: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Interpolate only bounded, fully missing hourly rows up to a fixed limit.

    Existing database-derived hourly rows are never overwritten. Gaps longer
    than ``max_gap_hours`` remain genuine gaps. No extrapolation is performed.
    """

    if max_gap_hours < 1:
        output = hourly.copy()
        output["is_imputed_hour"] = False
        output["imputed_gap_size_hours"] = 0
        output["imputation_method"] = None
        return output, []

    parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for hive_id, source_group in hourly.groupby(hive_column, sort=True):
        group = (
            source_group.sort_values(timestamp_column)
            .drop_duplicates(subset=[timestamp_column], keep="last")
            .copy()
        )

        first_hour = pd.Timestamp(group[timestamp_column].min())
        latest_hour = pd.Timestamp(group[timestamp_column].max())
        full_index = pd.date_range(first_hour, latest_hour, freq="h")

        indexed = group.set_index(timestamp_column)
        expanded = indexed.reindex(full_index)
        expanded.index.name = timestamp_column

        original_mask = expanded.index.isin(indexed.index)
        inserted_mask = ~original_mask

        expanded[hive_column] = str(hive_id)
        if "split" in expanded.columns:
            expanded["split"] = expanded["split"].ffill().bfill()
        if "_live_target_placeholder" in expanded.columns:
            expanded["_live_target_placeholder"] = expanded["_live_target_placeholder"].fillna(0)

        expanded["is_imputed_hour"] = False
        expanded["imputed_gap_size_hours"] = 0
        expanded["imputation_method"] = None

        eligible_rows = pd.Series(False, index=expanded.index)
        inserted_series = pd.Series(inserted_mask, index=expanded.index)
        run_ids = inserted_series.ne(inserted_series.shift()).cumsum()

        imputed_runs = 0
        imputed_hours = 0
        rejected_runs = 0

        for _, run in inserted_series.groupby(run_ids):
            if not bool(run.iloc[0]):
                continue

            gap_index = run.index
            gap_size = len(gap_index)
            before = gap_index[0] - pd.Timedelta(hours=1)
            after = gap_index[-1] + pd.Timedelta(hours=1)

            bounded = before in expanded.index and after in expanded.index
            within_limit = gap_size <= max_gap_hours
            boundaries_complete = bool(
                bounded
                and expanded.loc[
                    [before, after],
                    required_sensor_columns,
                ]
                .notna()
                .all(axis=None)
            )

            if not (bounded and within_limit and boundaries_complete):
                rejected_runs += 1
                continue

            for column in sensor_columns:
                if column not in expanded.columns:
                    continue

                start_value = expanded.at[before, column]
                end_value = expanded.at[after, column]
                if pd.isna(start_value) or pd.isna(end_value):
                    continue

                values = np.linspace(
                    float(start_value),
                    float(end_value),
                    gap_size + 2,
                )[1:-1]
                expanded.loc[gap_index, column] = values

            required_complete = expanded.loc[gap_index, required_sensor_columns].notna().all(axis=1)
            completed_index = required_complete.index[required_complete]
            if len(completed_index) != gap_size:
                rejected_runs += 1
                expanded.loc[gap_index, sensor_columns] = np.nan
                continue

            eligible_rows.loc[gap_index] = True
            expanded.loc[gap_index, "is_imputed_hour"] = True
            expanded.loc[
                gap_index,
                "imputed_gap_size_hours",
            ] = gap_size
            expanded.loc[
                gap_index,
                "imputation_method",
            ] = "bounded_linear_hourly"

            if "readings_in_hour" in expanded.columns:
                expanded.loc[gap_index, "readings_in_hour"] = 0

            count_columns = [
                column for column in expanded.columns if column.endswith("_reading_count")
            ]
            if count_columns:
                expanded.loc[gap_index, count_columns] = 0

            imputed_runs += 1
            imputed_hours += gap_size

        keep_mask = pd.Series(original_mask, index=expanded.index) | eligible_rows
        completed = expanded.loc[keep_mask].reset_index()
        parts.append(completed)

        summaries.append(
            {
                "hive_id": str(hive_id),
                "interpolation_enabled": True,
                "maximum_gap_hours": int(max_gap_hours),
                "imputed_gap_count": int(imputed_runs),
                "imputed_hourly_rows": int(imputed_hours),
                "rejected_gap_count": int(rejected_runs),
            }
        )

    if not parts:
        output = hourly.copy()
        output["is_imputed_hour"] = False
        output["imputed_gap_size_hours"] = 0
        output["imputation_method"] = None
        return output, summaries

    output = (
        pd.concat(parts, ignore_index=True)
        .sort_values([hive_column, timestamp_column])
        .reset_index(drop=True)
    )
    return output, summaries
