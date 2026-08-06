from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_FORECAST_STABILITY_REFERENCE: dict[str, float] = {
    "residual_rmse_scale": 3.0,
    "step_change_std_scale": 2.0,
    "calibration_quantile": 0.90,
}


def _as_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("A trajectory matrix must contain at least two time points")
    return matrix


def trajectory_diagnostics(
    current_scores: Any,
    future_scores: Any,
    *,
    step_hours: float = 1.0,
) -> dict[str, np.ndarray]:
    """Measure direction and smoothness of current-to-future score trajectories.

    BHSI and RoD answer different questions:
    - RoD is the fitted linear slope in score points per hour.
    - Forecast BHSI measures departures from a smooth trend. A smooth decline can
      therefore have high stability while RoD correctly reports deterioration.
    """

    future = _as_matrix(future_scores)
    current = np.asarray(current_scores, dtype=float).reshape(-1)
    if current.size == 1 and future.shape[0] > 1:
        current = np.repeat(current, future.shape[0])
    if current.size != future.shape[0]:
        raise ValueError("current_scores and future_scores must contain the same rows")

    trajectory = np.column_stack([current, future])
    x = np.arange(trajectory.shape[1], dtype=float) * float(step_hours)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered**2))
    y_mean = trajectory.mean(axis=1, keepdims=True)
    slopes = ((trajectory - y_mean) @ x_centered) / max(denominator, 1e-12)
    fitted = y_mean + slopes[:, None] * x_centered[None, :]
    residual_rmse = np.sqrt(np.mean((trajectory - fitted) ** 2, axis=1))

    changes_per_hour = np.diff(trajectory, axis=1) / max(float(step_hours), 1e-12)
    step_change_std = np.std(changes_per_hour, axis=1, ddof=0)
    score_range = np.ptp(trajectory, axis=1)

    return {
        "trajectory": trajectory,
        "rod_points_per_hour": slopes,
        "residual_rmse": residual_rmse,
        "step_change_std": step_change_std,
        "score_range": score_range,
    }


def calibrate_forecast_stability_reference(
    current_scores: Any,
    future_scores: Any,
    *,
    quantile: float = 0.90,
) -> dict[str, float]:
    """Calibrate BHSI scaling from actual future trajectories in training hives only."""

    diagnostics = trajectory_diagnostics(current_scores, future_scores)
    q = float(np.clip(quantile, 0.50, 0.99))
    residual_scale = float(np.quantile(diagnostics["residual_rmse"], q))
    step_scale = float(np.quantile(diagnostics["step_change_std"], q))
    return {
        "residual_rmse_scale": max(residual_scale, 0.25),
        "step_change_std_scale": max(step_scale, 0.25),
        "calibration_quantile": q,
    }


def forecast_bhsi(
    current_scores: Any,
    future_scores: Any,
    *,
    reference: dict[str, float] | None = None,
) -> np.ndarray:
    """Return a 0–100 stability index for the predicted health-score trajectory.

    The index is high when the path follows a smooth trend and low when the path
    fluctuates around that trend. Direction is deliberately excluded and is reported
    separately by Forecast RoD.
    """

    diagnostics = trajectory_diagnostics(current_scores, future_scores)
    ref = {**DEFAULT_FORECAST_STABILITY_REFERENCE, **(reference or {})}
    residual_ratio = diagnostics["residual_rmse"] / max(
        float(ref["residual_rmse_scale"]), 1e-6
    )
    step_ratio = diagnostics["step_change_std"] / max(
        float(ref["step_change_std_scale"]), 1e-6
    )
    instability = 0.60 * residual_ratio + 0.40 * step_ratio
    return np.clip(100.0 * np.exp(-instability), 0.0, 100.0)


def forecast_rod(
    current_scores: Any,
    future_scores: Any,
    *,
    step_hours: float = 1.0,
) -> np.ndarray:
    return trajectory_diagnostics(
        current_scores,
        future_scores,
        step_hours=step_hours,
    )["rod_points_per_hour"]


def interpolate_forecast_trajectory(
    *,
    current_score: float,
    hourly_scores: Iterable[float],
    anchor_timestamp: Any,
    resolution_minutes: int = 10,
) -> list[dict[str, Any]]:
    """Create a display trajectory without pretending that interpolation is model output.

    Native model points remain hourly. Intermediate ten-minute values are linear
    interpolation for dashboard continuity and are marked ``is_native_model_point=False``.
    """

    scores = np.asarray(list(hourly_scores), dtype=float)
    if scores.size < 1:
        return []
    resolution = max(1, int(resolution_minutes))
    horizon_minutes = int(scores.size * 60)
    minute_offsets = np.arange(0, horizon_minutes + 1, resolution, dtype=int)
    if minute_offsets[-1] != horizon_minutes:
        minute_offsets = np.append(minute_offsets, horizon_minutes)

    native_minutes = np.arange(0, horizon_minutes + 1, 60, dtype=float)
    native_scores = np.concatenate([[float(current_score)], scores])
    display_scores = np.interp(minute_offsets.astype(float), native_minutes, native_scores)

    anchor = pd.Timestamp(anchor_timestamp)
    if anchor.tzinfo is None:
        anchor = anchor.tz_localize("UTC")

    return [
        {
            "offset_minutes": int(minutes),
            "horizon_hours": float(minutes / 60.0),
            "forecast_timestamp": (
                anchor + pd.Timedelta(minutes=int(minutes))
            ).isoformat(),
            "score": float(np.clip(score, 1.0, 100.0)),
            "is_native_model_point": bool(minutes % 60 == 0),
            "value_kind": (
                "current_observation"
                if minutes == 0
                else "native_hourly_model_output"
                if minutes % 60 == 0
                else "display_interpolation"
            ),
        }
        for minutes, score in zip(minute_offsets, display_scores, strict=True)
    ]


def indicator_metrics(
    *,
    current_scores: Any,
    actual_future_scores: Any,
    predicted_future_scores: Any,
    reference: dict[str, float],
    stability_classifier,
    trend_classifier,
) -> dict[str, float]:
    actual_bhsi = forecast_bhsi(
        current_scores,
        actual_future_scores,
        reference=reference,
    )
    predicted_bhsi = forecast_bhsi(
        current_scores,
        predicted_future_scores,
        reference=reference,
    )
    actual_rod = forecast_rod(current_scores, actual_future_scores)
    predicted_rod = forecast_rod(current_scores, predicted_future_scores)

    actual_stability = np.asarray(
        [stability_classifier(value) for value in actual_bhsi],
        dtype=object,
    )
    predicted_stability = np.asarray(
        [stability_classifier(value) for value in predicted_bhsi],
        dtype=object,
    )
    actual_trend = np.asarray(
        [trend_classifier(value) for value in actual_rod],
        dtype=object,
    )
    predicted_trend = np.asarray(
        [trend_classifier(value) for value in predicted_rod],
        dtype=object,
    )

    return {
        "forecast_bhsi_mae": float(np.mean(np.abs(actual_bhsi - predicted_bhsi))),
        "forecast_bhsi_rmse": float(
            np.sqrt(np.mean((actual_bhsi - predicted_bhsi) ** 2))
        ),
        "forecast_bhsi_level_accuracy": float(
            np.mean(actual_stability == predicted_stability)
        ),
        "forecast_rod_mae": float(np.mean(np.abs(actual_rod - predicted_rod))),
        "forecast_rod_rmse": float(
            np.sqrt(np.mean((actual_rod - predicted_rod) ** 2))
        ),
        "forecast_trend_accuracy": float(np.mean(actual_trend == predicted_trend)),
    }


__all__ = [
    "DEFAULT_FORECAST_STABILITY_REFERENCE",
    "calibrate_forecast_stability_reference",
    "forecast_bhsi",
    "forecast_rod",
    "indicator_metrics",
    "interpolate_forecast_trajectory",
    "trajectory_diagnostics",
]
