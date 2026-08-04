from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_alert_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    beta: float,
    maximum_alert_fraction: float,
) -> dict[str, float | bool]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if thresholds.size == 0:
        return {
            "threshold": 0.5,
            "fbeta": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "alert_fraction": 0.0,
            "constraint_satisfied": False,
        }

    sorted_probability = np.sort(probability)
    candidates: list[dict[str, float | bool]] = []
    for index, threshold in enumerate(thresholds):
        denominator = beta**2 * precision[index] + recall[index]
        fbeta = (
            (1 + beta**2) * precision[index] * recall[index] / denominator
            if denominator > 0
            else 0.0
        )
        predicted_count = len(probability) - np.searchsorted(
            sorted_probability, threshold, side="left"
        )
        alert_fraction = predicted_count / len(probability) if len(probability) else 0.0
        candidates.append(
            {
                "threshold": float(threshold),
                "fbeta": float(fbeta),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "alert_fraction": float(alert_fraction),
                "constraint_satisfied": bool(alert_fraction <= maximum_alert_fraction),
            }
        )

    rank = lambda item: (
        item["fbeta"],
        item["recall"],
        item["precision"],
        item["threshold"],
    )
    best_overall = max(candidates, key=rank)
    feasible = [item for item in candidates if item["constraint_satisfied"]]
    if feasible:
        best_feasible = max(feasible, key=rank)
        # A zero-recall threshold is not a useful early-warning operating point.
        # Fall back to the best unconstrained F-beta threshold and expose that the
        # alert-fraction preference could not be satisfied.
        if best_feasible["fbeta"] > 0:
            return best_feasible
    return best_overall


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()

    result: dict[str, Any] = {
        "records": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "positive_rate": round(float(y_true.mean()), 8) if len(y_true) else None,
        "threshold": round(float(threshold), 8),
        "accuracy": round(float(accuracy_score(y_true, prediction)), 6),
        "precision": round(float(precision_score(y_true, prediction, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, prediction, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, prediction, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_true, prediction, zero_division=0)), 6),
        "f2": round(float(fbeta_score(y_true, prediction, beta=2, zero_division=0)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, prediction)), 6),
        "mae": round(float(mean_absolute_error(y_true, probability)), 8),
        "rmse": round(float(mean_squared_error(y_true, probability) ** 0.5), 8),
        "brier_score": round(float(brier_score_loss(y_true, probability)), 8),
        "alert_fraction": round(float(prediction.mean()), 8),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    result["pr_auc"] = _safe_metric(average_precision_score, y_true, probability)
    result["roc_auc"] = _safe_metric(roc_auc_score, y_true, probability)
    return result


def selection_score(metrics: dict[str, Any], event_metrics: dict[str, Any]) -> float:
    return round(
        0.45 * float(metrics.get("pr_auc") or 0.0)
        + 0.25 * float(metrics.get("f2") or 0.0)
        + 0.20 * float(event_metrics.get("event_recall") or 0.0)
        + 0.10 * float(metrics.get("precision") or 0.0),
        8,
    )


def _safe_metric(function, y_true: np.ndarray, probability: np.ndarray) -> float | None:
    try:
        return round(float(function(y_true, probability)), 6)
    except ValueError:
        return None
