from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .features import TARGET_COLUMN, normalise_historical


def binary_target_persistence_audit(
    frame: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 6, 24),
) -> dict[str, Any]:
    """Quantify why row-level future binary status can produce misleading accuracy.

    The original task predicted ``brood_health_healthy_1`` at a nearby future hour.
    Long healthy/unhealthy episodes make a persistence rule (future = current) look
    excellent even when it provides little advance warning of deterioration.
    """

    data = normalise_historical(frame)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Historical data do not contain {TARGET_COLUMN}")

    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    valid_target = target.dropna()
    rows: list[dict[str, Any]] = []
    for raw_horizon in horizons:
        horizon = int(raw_horizon)
        if horizon < 1:
            raise ValueError("Persistence-audit horizons must be positive integers")
        future = target.groupby(data["hive_id"], sort=False).shift(-horizon)
        comparable = target.notna() & future.notna()
        compared = int(comparable.sum())
        same = int((target[comparable] == future[comparable]).sum())
        transitions = compared - same
        rows.append(
            {
                "horizon_hours": horizon,
                "comparable_rows": compared,
                "same_status_rows": same,
                "transition_rows": transitions,
                "persistence_accuracy": float(same / compared) if compared else None,
                "transition_rate": float(transitions / compared) if compared else None,
            }
        )

    healthy_rows = int((valid_target == 1).sum())
    unhealthy_rows = int((valid_target == 0).sum())
    total = len(valid_target)
    return {
        "target_column": TARGET_COLUMN,
        "valid_rows": total,
        "healthy_rows": healthy_rows,
        "unhealthy_rows": unhealthy_rows,
        "healthy_rate": float(healthy_rows / total) if total else None,
        "unhealthy_rate": float(unhealthy_rows / total) if total else None,
        "horizons": rows,
        "interpretation": (
            "Persistence accuracy answers whether a nearby future binary label stays the same. "
            "It is not sufficient evidence of early-warning performance when transitions are rare."
        ),
    }


def feature_leakage_audit(feature_columns: Iterable[str]) -> dict[str, Any]:
    """Inspect the generated model schema instead of reporting hard-coded claims."""

    columns = [str(column) for column in feature_columns]
    lowered = {column: column.lower() for column in columns}

    target_like = sorted(
        column
        for column, name in lowered.items()
        if any(
            fragment in name
            for fragment in ("brood_health_healthy", "target", "label", "observed_healthy")
        )
    )
    future_like = sorted(
        column
        for column, name in lowered.items()
        if any(fragment in name for fragment in ("future_", "lead_", "next_"))
    )
    hive_identifiers = sorted(
        column
        for column, name in lowered.items()
        if name in {"hive", "hive_id", "device", "device_id"}
    )
    absolute_time = sorted(
        column
        for column, name in lowered.items()
        if name in {"timestamp", "date", "year", "month", "day_of_year"} or "day_of_year" in name
    )
    absolute_weight = sorted(
        column
        for column, name in lowered.items()
        if name in {"weight", "weight_kg", "total_weight"}
    )

    passed = not any((target_like, future_like, hive_identifiers, absolute_time, absolute_weight))
    return {
        "passed": passed,
        "target_columns_in_features": target_like,
        "future_sensor_values_in_features": future_like,
        "hive_identifier_features": hive_identifiers,
        "absolute_time_features": absolute_time,
        "absolute_weight_features": absolute_weight,
        "current_or_lagged_binary_target_used": bool(target_like),
        "future_sensor_values_used_as_features": bool(future_like),
        "hive_id_used_as_feature": bool(hive_identifiers),
        "absolute_date_or_day_of_year_used_as_feature": bool(absolute_time),
        "absolute_weight_used_as_feature": bool(absolute_weight),
    }
