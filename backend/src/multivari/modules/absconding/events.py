from __future__ import annotations

from typing import Any

import pandas as pd

from multivari.common.schema import HIVE_COLUMN, TIMESTAMP_COLUMN


def build_event_episodes(
    df: pd.DataFrame,
    *,
    event_column: str,
    merge_gap_hours: int,
) -> pd.DataFrame:
    markers = df.loc[df[event_column].eq(1), [HIVE_COLUMN, TIMESTAMP_COLUMN]].copy()
    markers[TIMESTAMP_COLUMN] = pd.to_datetime(markers[TIMESTAMP_COLUMN])
    markers = markers.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN])
    episodes: list[dict[str, Any]] = []

    for hive_id, group in markers.groupby(HIVE_COLUMN, sort=False):
        current: list[pd.Timestamp] = []
        previous: pd.Timestamp | None = None
        episode_number = 0
        for timestamp in group[TIMESTAMP_COLUMN]:
            if previous is None or (timestamp - previous).total_seconds() / 3600 <= merge_gap_hours:
                current.append(timestamp)
            else:
                episode_number += 1
                episodes.append(_episode_row(str(hive_id), episode_number, current))
                current = [timestamp]
            previous = timestamp
        if current:
            episode_number += 1
            episodes.append(_episode_row(str(hive_id), episode_number, current))

    return pd.DataFrame(
        episodes,
        columns=["episode_id", "hive_id", "event_start", "event_end", "marker_count"],
    )


def _episode_row(hive_id: str, episode_number: int, markers: list[pd.Timestamp]) -> dict[str, Any]:
    return {
        "episode_id": f"{hive_id}-absconding-{episode_number:02d}",
        "hive_id": hive_id,
        "event_start": min(markers),
        "event_end": max(markers),
        "marker_count": len(markers),
    }


def attach_episode_splits(episodes: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        result = episodes.copy()
        result["split"] = pd.Series(dtype="string")
        return result
    split_lookup = prepared[[HIVE_COLUMN, TIMESTAMP_COLUMN, "split"]].drop_duplicates()
    result = episodes.merge(
        split_lookup,
        left_on=["hive_id", "event_start"],
        right_on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        how="left",
    ).drop(columns=[TIMESTAMP_COLUMN])
    return result


def evaluate_event_warnings(
    predictions: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    threshold: float,
    horizon_hours: int,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = episodes.loc[episodes["split"].eq(split)].copy() if not episodes.empty else episodes
    details: list[dict[str, Any]] = []

    for row in selected.itertuples(index=False):
        start = pd.Timestamp(row.event_start)
        lower = start - pd.Timedelta(hours=horizon_hours)
        window = predictions.loc[
            predictions[HIVE_COLUMN].eq(row.hive_id)
            & predictions[TIMESTAMP_COLUMN].ge(lower)
            & predictions[TIMESTAMP_COLUMN].lt(start)
            & predictions["split"].eq(split)
        ].sort_values(TIMESTAMP_COLUMN)
        alerted = window.loc[window["absconding_probability"].ge(threshold)]
        first_alert = alerted[TIMESTAMP_COLUMN].min() if not alerted.empty else None
        lead_hours = (
            round((start - first_alert).total_seconds() / 3600, 2)
            if first_alert is not None
            else None
        )
        details.append(
            {
                "episode_id": row.episode_id,
                "hive_id": row.hive_id,
                "event_start": start.isoformat(),
                "marker_count": int(row.marker_count),
                "detected": bool(first_alert is not None),
                "first_alert": first_alert.isoformat() if first_alert is not None else None,
                "lead_hours": lead_hours,
                "maximum_probability": round(
                    float(window["absconding_probability"].max()), 6
                )
                if not window.empty
                else None,
                "warning_rows": int(len(alerted)),
            }
        )

    detected = sum(item["detected"] for item in details)
    leads = [item["lead_hours"] for item in details if item["lead_hours"] is not None]
    metrics = {
        "event_count": len(details),
        "detected_event_count": detected,
        "event_recall": round(detected / len(details), 6) if details else None,
        "median_lead_hours": round(float(pd.Series(leads).median()), 2) if leads else None,
    }
    return metrics, details
