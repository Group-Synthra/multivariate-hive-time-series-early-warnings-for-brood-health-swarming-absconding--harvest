from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HIVE_COLUMN = "hive_id"
TIMESTAMP_COLUMN = "timestamp"
SPLIT_COLUMN = "split"
EVENT_ID_COLUMN = "harvest_event_id"
SESSION_ID_COLUMN = "harvest_session_id"


@dataclass(frozen=True)
class CandidateResult:
    model_name: str
    feature_set_name: str
    feature_columns: list[str]
    estimator: BaseEstimator
    validation_probabilities: np.ndarray
    validation_metrics: dict[str, Any]
    threshold: float
    threshold_sweep: pd.DataFrame
    validation_event_detection: pd.DataFrame
    training_seconds: float
    training_rows: int
    training_positive_rows: int
    training_negative_rows: int


def _resolve_path(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    frame_name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: {missing}"
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=_json_safe,
        ),
        encoding="utf-8",
    )


def cluster_harvest_sessions(
    events: pd.DataFrame,
    *,
    session_gap_hours: int,
) -> pd.DataFrame:
    """
    Group close events across hives into apiary-level harvest sessions.

    The first event starts a session. A new session starts only when the gap
    from the previous event is greater than the configured number of hours.
    """
    if session_gap_hours < 0:
        raise ValueError("session_gap_hours cannot be negative")

    _require_columns(
        events,
        {
            HIVE_COLUMN,
            EVENT_ID_COLUMN,
            "event_start",
            SPLIT_COLUMN,
        },
        frame_name="Reviewed events",
    )

    result = events.copy()
    result["event_start"] = pd.to_datetime(
        result["event_start"],
        errors="raise",
    )
    result = result.sort_values("event_start").reset_index(drop=True)

    gap = result["event_start"].diff().dt.total_seconds().div(3600)
    starts_new = gap.isna() | gap.gt(session_gap_hours)
    session_number = starts_new.cumsum().astype(int)
    result[SESSION_ID_COLUMN] = session_number.map(
        lambda value: f"harvest_session_{value:03d}"
    )
    return result


def attach_future_event_metadata(
    rows: pd.DataFrame,
    events: pd.DataFrame,
    *,
    target_column: str,
    horizon_hours: int,
) -> pd.DataFrame:
    """
    Attach the next reviewed event and its session to positive rows.

    The current timestamp is excluded. This mirrors the target definition:
    an event must start in (t, t + horizon].
    """
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be greater than zero")

    _require_columns(
        rows,
        {
            HIVE_COLUMN,
            TIMESTAMP_COLUMN,
            target_column,
        },
        frame_name="Feature rows",
    )
    _require_columns(
        events,
        {
            HIVE_COLUMN,
            EVENT_ID_COLUMN,
            SESSION_ID_COLUMN,
            "event_start",
        },
        frame_name="Session-labelled events",
    )

    result = rows.copy()
    result[TIMESTAMP_COLUMN] = pd.to_datetime(
        result[TIMESTAMP_COLUMN],
        errors="raise",
    )
    result[EVENT_ID_COLUMN] = pd.NA
    result[SESSION_ID_COLUMN] = pd.NA
    result["matched_event_start"] = pd.NaT

    event_lookup: dict[str, pd.DataFrame] = {}
    for hive_id, group in events.groupby(HIVE_COLUMN, sort=False):
        event_lookup[str(hive_id)] = (
            group.sort_values("event_start")
            .reset_index(drop=True)
        )

    positive_mask = result[target_column].eq(1)
    positive_rows = result.loc[positive_mask]

    for hive_id, group in positive_rows.groupby(
        HIVE_COLUMN,
        sort=False,
    ):
        hive_events = event_lookup.get(str(hive_id))
        if hive_events is None or hive_events.empty:
            raise ValueError(
                f"Positive rows exist for hive {hive_id}, "
                "but no reviewed event is available."
            )

        event_times = hive_events["event_start"].to_numpy(
            dtype="datetime64[ns]"
        )
        row_times = group[TIMESTAMP_COLUMN].to_numpy(
            dtype="datetime64[ns]"
        )
        next_positions = np.searchsorted(
            event_times,
            row_times,
            side="right",
        )

        valid = next_positions < len(event_times)
        if not valid.all():
            raise ValueError(
                f"Some positive rows for hive {hive_id} have "
                "no future reviewed event."
            )

        matched_times = event_times[next_positions]
        hours_to_event = (
            matched_times - row_times
        ) / np.timedelta64(1, "h")

        valid_horizon = (
            (hours_to_event > 0)
            & (hours_to_event <= horizon_hours)
        )
        if not valid_horizon.all():
            raise ValueError(
                f"Positive rows for hive {hive_id} do not match "
                "the configured future horizon."
            )

        matched = hive_events.iloc[next_positions]
        result.loc[group.index, EVENT_ID_COLUMN] = (
            matched[EVENT_ID_COLUMN].to_numpy()
        )
        result.loc[group.index, SESSION_ID_COLUMN] = (
            matched[SESSION_ID_COLUMN].to_numpy()
        )
        result.loc[group.index, "matched_event_start"] = (
            matched["event_start"].to_numpy()
        )

    return result


def build_feature_sets(
    all_features: list[str],
    feature_set_config: dict[str, Any],
) -> dict[str, list[str]]:
    available = list(dict.fromkeys(all_features))
    available_set = set(available)
    result: dict[str, list[str]] = {}

    for name, settings in feature_set_config.items():
        if settings.get("include_all"):
            selected = available.copy()
        elif "include" in settings:
            requested = [str(value) for value in settings["include"]]
            missing = sorted(set(requested).difference(available_set))
            if missing:
                raise ValueError(
                    f"Feature set '{name}' requests missing features: "
                    f"{missing}"
                )
            selected = requested
        else:
            include_prefixes = [
                str(value)
                for value in settings.get("include_prefixes", [])
            ]
            selected = [
                feature
                for feature in available
                if any(
                    feature.startswith(prefix)
                    for prefix in include_prefixes
                )
            ]

        exclude_prefixes = [
            str(value)
            for value in settings.get("exclude_prefixes", [])
        ]
        if exclude_prefixes:
            selected = [
                feature
                for feature in selected
                if not any(
                    feature.startswith(prefix)
                    for prefix in exclude_prefixes
                )
            ]

        selected = list(dict.fromkeys(selected))
        if not selected:
            raise ValueError(
                f"Feature set '{name}' contains no features."
            )
        result[name] = selected

    return result


def sample_training_rows(
    training_rows: pd.DataFrame,
    *,
    target_column: str,
    maximum_negative_to_positive_ratio: int,
    random_state: int,
) -> pd.DataFrame:
    if maximum_negative_to_positive_ratio <= 0:
        raise ValueError(
            "maximum_negative_to_positive_ratio must be positive"
        )

    positives = training_rows.loc[
        training_rows[target_column].eq(1)
    ]
    negatives = training_rows.loc[
        training_rows[target_column].eq(0)
    ]

    if positives.empty or negatives.empty:
        raise ValueError(
            "Training data must contain both positive and negative rows."
        )

    maximum_negatives = (
        len(positives) * maximum_negative_to_positive_ratio
    )
    if len(negatives) > maximum_negatives:
        negatives = negatives.sample(
            n=maximum_negatives,
            random_state=random_state,
            replace=False,
        )

    sampled = pd.concat(
        [positives, negatives],
        ignore_index=True,
    )
    return sampled.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)


def calculate_session_balanced_weights(
    rows: pd.DataFrame,
    *,
    target_column: str,
) -> np.ndarray:
    """
    Give positives and negatives equal total weight.

    Within positives, each harvest session receives equal weight. Within a
    session, each event receives equal weight, and rows inside an event share
    that event weight equally. This prevents one batch harvest involving many
    hives from dominating the fitted model.
    """
    _require_columns(
        rows,
        {
            target_column,
            EVENT_ID_COLUMN,
            SESSION_ID_COLUMN,
        },
        frame_name="Sampled training rows",
    )

    weights = np.zeros(len(rows), dtype=float)
    positive_mask = rows[target_column].eq(1).to_numpy()
    negative_mask = ~positive_mask

    positive_rows = rows.loc[positive_mask]
    negative_count = int(negative_mask.sum())

    sessions = positive_rows[SESSION_ID_COLUMN].dropna().unique()
    if len(sessions) == 0 or negative_count == 0:
        raise ValueError(
            "Session-balanced weights require positive sessions "
            "and negative rows."
        )

    positive_total_weight = 0.5
    negative_total_weight = 0.5
    session_weight = positive_total_weight / len(sessions)

    for session_id in sessions:
        session_rows = positive_rows.loc[
            positive_rows[SESSION_ID_COLUMN].eq(session_id)
        ]
        events = session_rows[EVENT_ID_COLUMN].dropna().unique()
        if len(events) == 0:
            raise ValueError(
                f"Session {session_id} contains no event IDs."
            )
        event_weight = session_weight / len(events)

        for event_id in events:
            event_indices = session_rows.loc[
                session_rows[EVENT_ID_COLUMN].eq(event_id)
            ].index
            per_row_weight = event_weight / len(event_indices)
            weights[event_indices.to_numpy()] = per_row_weight

    weights[np.flatnonzero(negative_mask)] = (
        negative_total_weight / negative_count
    )

    # Preserve the relative class/session/event weighting while
    # keeping the average sample weight equal to one. Normalizing the
    # complete vector to sum to one makes regularization dominate
    # Logistic Regression and prevents XGBoost/LightGBM tree splits
    # because their weighted Hessian totals become extremely small.
    weights *= len(rows)

    if not math.isclose(weights.mean(), 1.0, rel_tol=1e-9):
        raise RuntimeError(
            "Session-balanced sample weights do not have mean one."
        )

    return weights


def _make_estimator(
    model_name: str,
    settings: dict[str, Any],
    *,
    random_state: int,
) -> BaseEstimator:
    if model_name == "logistic_regression":
        model = LogisticRegression(
            C=float(settings["C"]),
            penalty="elasticnet",
            l1_ratio=float(settings["l1_ratio"]),
            solver="saga",
            max_iter=int(settings["max_iter"]),
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", model),
            ]
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(settings["n_estimators"]),
            max_depth=int(settings["max_depth"]),
            min_samples_leaf=int(
                settings["min_samples_leaf"]
            ),
            max_features=settings["max_features"],
            max_samples=float(settings["max_samples"]),
            n_jobs=-1,
            random_state=random_state,
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "XGBoost is not installed. Run: pip install xgboost"
            ) from error

        return XGBClassifier(
            n_estimators=int(settings["n_estimators"]),
            learning_rate=float(settings["learning_rate"]),
            max_depth=int(settings["max_depth"]),
            min_child_weight=float(
                settings["min_child_weight"]
            ),
            subsample=float(settings["subsample"]),
            colsample_bytree=float(
                settings["colsample_bytree"]
            ),
            reg_alpha=float(settings["reg_alpha"]),
            reg_lambda=float(settings["reg_lambda"]),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError(
                "LightGBM is not installed. Run: pip install lightgbm"
            ) from error

        return LGBMClassifier(
            n_estimators=int(settings["n_estimators"]),
            learning_rate=float(settings["learning_rate"]),
            num_leaves=int(settings["num_leaves"]),
            max_depth=int(settings["max_depth"]),
            min_child_samples=int(
                settings["min_child_samples"]
            ),
            subsample=float(settings["subsample"]),
            colsample_bytree=float(
                settings["colsample_bytree"]
            ),
            reg_alpha=float(settings["reg_alpha"]),
            reg_lambda=float(settings["reg_lambda"]),
            objective="binary",
            n_jobs=-1,
            random_state=random_state,
            verbosity=-1,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def _fit_estimator(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    weights: np.ndarray,
) -> None:
    if isinstance(estimator, Pipeline):
        estimator.fit(
            features,
            target,
            model__sample_weight=weights,
        )
    else:
        estimator.fit(
            features,
            target,
            sample_weight=weights,
        )


def _positive_probabilities(
    estimator: BaseEstimator,
    features: pd.DataFrame,
) -> np.ndarray:
    probabilities = estimator.predict_proba(features)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            "Classifier predict_proba must return two columns."
        )
    return probabilities[:, 1].astype(float)


def calculate_row_metrics(
    target: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(target, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "pr_auc": float(
            average_precision_score(y_true, y_prob)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier_score": float(
            brier_score_loss(y_true, y_prob)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def count_false_alert_episodes(
    predictions: pd.DataFrame,
    *,
    target_column: str,
    probability_column: str,
    threshold: float,
    gap_hours: int,
) -> int:
    false_alerts = predictions.loc[
        predictions[target_column].eq(0)
        & predictions[probability_column].ge(threshold)
    ].copy()

    if false_alerts.empty:
        return 0

    false_alerts = false_alerts.sort_values(
        [HIVE_COLUMN, TIMESTAMP_COLUMN]
    )
    elapsed = (
        false_alerts.groupby(HIVE_COLUMN)[TIMESTAMP_COLUMN]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )
    starts_episode = elapsed.isna() | elapsed.gt(gap_hours)
    return int(starts_episode.sum())


def build_event_detection_table(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    split: str,
    probability_column: str,
    threshold: float,
    horizon_hours: int,
) -> pd.DataFrame:
    split_events = events.loc[
        events[SPLIT_COLUMN].eq(split)
    ].sort_values("event_start")

    records: list[dict[str, Any]] = []
    horizon = pd.Timedelta(hours=horizon_hours)

    for event in split_events.itertuples(index=False):
        event_start = event.event_start
        event_rows = predictions.loc[
            predictions[HIVE_COLUMN].eq(event.hive_id)
            & predictions[TIMESTAMP_COLUMN].ge(
                event_start - horizon
            )
            & predictions[TIMESTAMP_COLUMN].lt(event_start)
        ].sort_values(TIMESTAMP_COLUMN)

        alert_rows = event_rows.loc[
            event_rows[probability_column].ge(threshold)
        ]
        detected = not alert_rows.empty

        first_alert_time = (
            alert_rows[TIMESTAMP_COLUMN].iloc[0]
            if detected
            else pd.NaT
        )
        lead_hours = (
            (
                event_start - first_alert_time
            ).total_seconds()
            / 3600
            if detected
            else np.nan
        )

        records.append(
            {
                EVENT_ID_COLUMN: event.harvest_event_id,
                SESSION_ID_COLUMN: event.harvest_session_id,
                HIVE_COLUMN: event.hive_id,
                SPLIT_COLUMN: split,
                "event_start": event_start,
                "available_prediction_rows": len(event_rows),
                "alert_rows": len(alert_rows),
                "detected": bool(detected),
                "first_alert_time": first_alert_time,
                "lead_hours": lead_hours,
                "maximum_probability": (
                    float(event_rows[probability_column].max())
                    if not event_rows.empty
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(records)


def select_operating_threshold(
    validation_predictions: pd.DataFrame,
    validation_events: pd.DataFrame,
    *,
    target_column: str,
    probability_column: str,
    horizon_hours: int,
    false_alert_gap_hours: int,
    threshold_grid_points: int,
    minimum_event_recall: float,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    probabilities = validation_predictions[
        probability_column
    ].to_numpy(dtype=float)

    quantiles = np.quantile(
        probabilities,
        np.linspace(0.0, 1.0, min(101, len(probabilities))),
    )
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.001,
                    0.999,
                    threshold_grid_points,
                ),
                quantiles,
            ]
        )
    )

    records: list[dict[str, Any]] = []
    detection_by_threshold: dict[
        float,
        pd.DataFrame,
    ] = {}

    for threshold in thresholds:
        row_metrics = calculate_row_metrics(
            validation_predictions[target_column],
            probabilities,
            threshold=float(threshold),
        )
        event_detection = build_event_detection_table(
            validation_predictions,
            validation_events,
            split="validation",
            probability_column=probability_column,
            threshold=float(threshold),
            horizon_hours=horizon_hours,
        )
        event_recall = (
            float(event_detection["detected"].mean())
            if not event_detection.empty
            else 0.0
        )
        false_alert_episodes = count_false_alert_episodes(
            validation_predictions,
            target_column=target_column,
            probability_column=probability_column,
            threshold=float(threshold),
            gap_hours=false_alert_gap_hours,
        )

        record = {
            "threshold": float(threshold),
            **row_metrics,
            "event_recall": event_recall,
            "false_alert_episodes": false_alert_episodes,
        }
        records.append(record)
        detection_by_threshold[float(threshold)] = (
            event_detection
        )

    sweep = pd.DataFrame(records)
    eligible = sweep.loc[
        sweep["event_recall"].ge(minimum_event_recall)
    ]

    if eligible.empty:
        ranked = sweep.sort_values(
            [
                "event_recall",
                "f1",
                "precision",
                "false_alert_episodes",
                "threshold",
            ],
            ascending=[False, False, False, True, False],
        )
    else:
        ranked = eligible.sort_values(
            [
                "f1",
                "precision",
                "false_alert_episodes",
                "threshold",
            ],
            ascending=[False, False, True, False],
        )

    selected_threshold = float(ranked.iloc[0]["threshold"])
    return (
        selected_threshold,
        sweep.sort_values("threshold").reset_index(drop=True),
        detection_by_threshold[selected_threshold],
    )


def _candidate_selection_key(
    candidate: CandidateResult,
    *,
    minimum_event_recall: float,
) -> tuple[Any, ...]:
    event_recall = float(
        candidate.validation_metrics["event_recall"]
    )
    eligible = int(event_recall >= minimum_event_recall)

    model_complexity = {
        "logistic_regression": 0,
        "random_forest": 1,
        "xgboost": 2,
        "lightgbm": 3,
    }[candidate.model_name]

    return (
        eligible,
        float(candidate.validation_metrics["pr_auc"]),
        -int(
            candidate.validation_metrics[
                "false_alert_episodes"
            ]
        ),
        -len(candidate.feature_columns),
        -model_complexity,
    )


def _extract_feature_importance(
    estimator: BaseEstimator,
    feature_columns: list[str],
) -> pd.DataFrame:
    fitted = estimator
    if isinstance(estimator, Pipeline):
        fitted = estimator.named_steps["model"]

    if hasattr(fitted, "coef_"):
        values = np.asarray(fitted.coef_).reshape(-1)
        importance_type = "coefficient"
    elif hasattr(fitted, "feature_importances_"):
        values = np.asarray(
            fitted.feature_importances_
        ).reshape(-1)
        importance_type = "feature_importance"
    else:
        return pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "absolute_importance",
                "importance_type",
            ]
        )

    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": values,
            "absolute_importance": np.abs(values),
            "importance_type": importance_type,
        }
    )
    return result.sort_values(
        "absolute_importance",
        ascending=False,
    ).reset_index(drop=True)


def _run_grouped_hive_robustness(
    *,
    selected: CandidateResult,
    training_rows: pd.DataFrame,
    training_events: pd.DataFrame,
    model_settings: dict[str, Any],
    target_column: str,
    maximum_negative_to_positive_ratio: int,
    random_state: int,
    threshold: float,
    horizon_hours: int,
    false_alert_gap_hours: int,
) -> pd.DataFrame:
    positive_hives = sorted(
        training_events[HIVE_COLUMN].unique().tolist()
    )
    records: list[dict[str, Any]] = []

    for fold_number, held_out_hive in enumerate(
        positive_hives,
        start=1,
    ):
        fold_train = training_rows.loc[
            training_rows[HIVE_COLUMN].ne(held_out_hive)
        ].copy()
        fold_validation = training_rows.loc[
            training_rows[HIVE_COLUMN].eq(held_out_hive)
        ].copy()

        if (
            fold_train[target_column].nunique() < 2
            or fold_validation[target_column].nunique() < 2
        ):
            continue

        sampled = sample_training_rows(
            fold_train,
            target_column=target_column,
            maximum_negative_to_positive_ratio=(
                maximum_negative_to_positive_ratio
            ),
            random_state=random_state + fold_number,
        )
        weights = calculate_session_balanced_weights(
            sampled,
            target_column=target_column,
        )
        estimator = _make_estimator(
            selected.model_name,
            model_settings,
            random_state=random_state + fold_number,
        )
        _fit_estimator(
            estimator,
            sampled[selected.feature_columns],
            sampled[target_column],
            weights,
        )

        probabilities = _positive_probabilities(
            estimator,
            fold_validation[selected.feature_columns],
        )
        fold_predictions = fold_validation[
            [
                TIMESTAMP_COLUMN,
                HIVE_COLUMN,
                SPLIT_COLUMN,
                target_column,
            ]
        ].copy()
        fold_predictions["probability"] = probabilities

        row_metrics = calculate_row_metrics(
            fold_validation[target_column],
            probabilities,
            threshold=threshold,
        )
        fold_events = training_events.loc[
            training_events[HIVE_COLUMN].eq(held_out_hive)
        ].copy()
        fold_events[SPLIT_COLUMN] = "validation"
        fold_predictions[SPLIT_COLUMN] = "validation"
        event_detection = build_event_detection_table(
            fold_predictions,
            fold_events,
            split="validation",
            probability_column="probability",
            threshold=threshold,
            horizon_hours=horizon_hours,
        )

        records.append(
            {
                "fold": fold_number,
                "held_out_hive": held_out_hive,
                "held_out_events": len(fold_events),
                "training_rows": len(sampled),
                "validation_rows": len(fold_validation),
                "pr_auc": row_metrics["pr_auc"],
                "precision": row_metrics["precision"],
                "recall": row_metrics["recall"],
                "f1": row_metrics["f1"],
                "event_recall": (
                    float(event_detection["detected"].mean())
                    if not event_detection.empty
                    else 0.0
                ),
                "false_alert_episodes": (
                    count_false_alert_episodes(
                        fold_predictions,
                        target_column=target_column,
                        probability_column="probability",
                        threshold=threshold,
                        gap_hours=false_alert_gap_hours,
                    )
                ),
            }
        )

    return pd.DataFrame(records)


def run_research_model_comparison_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    settings = config["research_model_comparison"]

    feature_path = _resolve_path(
        root,
        settings["feature_dataset_path"],
    )
    event_path = _resolve_path(
        root,
        settings["reviewed_events_path"],
    )
    manifest_path = _resolve_path(
        root,
        settings["feature_manifest_path"],
    )
    output_directory = _resolve_path(
        root,
        settings["output_directory"],
    )
    model_directory = _resolve_path(
        root,
        settings["model_directory"],
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    target_column = str(settings["target_column"])
    horizon_hours = int(settings["horizon_hours"])
    random_state = int(settings["random_state"])
    session_gap_hours = int(settings["session_gap_hours"])
    false_alert_gap_hours = int(
        settings["false_alert_gap_hours"]
    )
    negative_ratio = int(
        settings["maximum_negative_to_positive_ratio"]
    )
    minimum_event_recall = float(
        settings["minimum_validation_event_recall"]
    )
    threshold_grid_points = int(
        settings["threshold_grid_points"]
    )

    features = pd.read_parquet(feature_path)
    events = pd.read_parquet(event_path)
    manifest = pd.read_csv(manifest_path)

    features[TIMESTAMP_COLUMN] = pd.to_datetime(
        features[TIMESTAMP_COLUMN],
        errors="raise",
    )
    events["event_start"] = pd.to_datetime(
        events["event_start"],
        errors="raise",
    )

    feature_columns = manifest["feature_name"].astype(str).tolist()
    _require_columns(
        features,
        {
            TIMESTAMP_COLUMN,
            HIVE_COLUMN,
            SPLIT_COLUMN,
            target_column,
            *feature_columns,
        },
        frame_name="Reviewed feature dataset",
    )

    session_events = cluster_harvest_sessions(
        events,
        session_gap_hours=session_gap_hours,
    )
    rows = attach_future_event_metadata(
        features,
        session_events,
        target_column=target_column,
        horizon_hours=horizon_hours,
    )

    train_rows = rows.loc[
        rows[SPLIT_COLUMN].eq("train")
    ].copy()
    validation_rows = rows.loc[
        rows[SPLIT_COLUMN].eq("validation")
    ].copy()
    test_rows = rows.loc[
        rows[SPLIT_COLUMN].eq("test")
    ].copy()

    if (
        train_rows.empty
        or validation_rows.empty
        or test_rows.empty
    ):
        raise ValueError(
            "Train, validation and test rows must all be present."
        )

    feature_sets = build_feature_sets(
        feature_columns,
        settings["feature_sets"],
    )

    candidates: list[CandidateResult] = []
    comparison_records: list[dict[str, Any]] = []

    for feature_set_name, selected_features in feature_sets.items():
        for model_name, model_settings in settings["models"].items():
            if not bool(model_settings.get("enabled", True)):
                comparison_records.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "status": "disabled",
                    }
                )
                continue

            try:
                sampled_train = sample_training_rows(
                    train_rows,
                    target_column=target_column,
                    maximum_negative_to_positive_ratio=negative_ratio,
                    random_state=random_state,
                )
                weights = calculate_session_balanced_weights(
                    sampled_train,
                    target_column=target_column,
                )
                estimator = _make_estimator(
                    model_name,
                    model_settings,
                    random_state=random_state,
                )

                started = time.perf_counter()
                _fit_estimator(
                    estimator,
                    sampled_train[selected_features],
                    sampled_train[target_column],
                    weights,
                )
                training_seconds = (
                    time.perf_counter() - started
                )

                validation_probabilities = (
                    _positive_probabilities(
                        estimator,
                        validation_rows[selected_features],
                    )
                )
                validation_predictions = validation_rows[
                    [
                        TIMESTAMP_COLUMN,
                        HIVE_COLUMN,
                        SPLIT_COLUMN,
                        target_column,
                    ]
                ].copy()
                validation_predictions["probability"] = (
                    validation_probabilities
                )

                threshold, sweep, event_detection = (
                    select_operating_threshold(
                        validation_predictions,
                        session_events,
                        target_column=target_column,
                        probability_column="probability",
                        horizon_hours=horizon_hours,
                        false_alert_gap_hours=(
                            false_alert_gap_hours
                        ),
                        threshold_grid_points=(
                            threshold_grid_points
                        ),
                        minimum_event_recall=(
                            minimum_event_recall
                        ),
                    )
                )
                row_metrics = calculate_row_metrics(
                    validation_rows[target_column],
                    validation_probabilities,
                    threshold=threshold,
                )
                event_recall = (
                    float(event_detection["detected"].mean())
                    if not event_detection.empty
                    else 0.0
                )
                false_alert_episodes = (
                    count_false_alert_episodes(
                        validation_predictions,
                        target_column=target_column,
                        probability_column="probability",
                        threshold=threshold,
                        gap_hours=false_alert_gap_hours,
                    )
                )

                validation_metrics = {
                    **row_metrics,
                    "event_recall": event_recall,
                    "false_alert_episodes": (
                        false_alert_episodes
                    ),
                }
                candidate = CandidateResult(
                    model_name=model_name,
                    feature_set_name=feature_set_name,
                    feature_columns=selected_features,
                    estimator=estimator,
                    validation_probabilities=(
                        validation_probabilities
                    ),
                    validation_metrics=validation_metrics,
                    threshold=threshold,
                    threshold_sweep=sweep,
                    validation_event_detection=(
                        event_detection
                    ),
                    training_seconds=training_seconds,
                    training_rows=len(sampled_train),
                    training_positive_rows=int(
                        sampled_train[target_column].sum()
                    ),
                    training_negative_rows=int(
                        len(sampled_train)
                        - sampled_train[target_column].sum()
                    ),
                )
                candidates.append(candidate)

                comparison_records.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "status": "ok",
                        "feature_count": len(selected_features),
                        "training_rows": len(sampled_train),
                        "training_positive_rows": int(
                            sampled_train[target_column].sum()
                        ),
                        "training_negative_rows": int(
                            len(sampled_train)
                            - sampled_train[target_column].sum()
                        ),
                        "training_seconds": training_seconds,
                        "validation_threshold": threshold,
                        "validation_pr_auc": row_metrics[
                            "pr_auc"
                        ],
                        "validation_roc_auc": row_metrics[
                            "roc_auc"
                        ],
                        "validation_brier_score": row_metrics[
                            "brier_score"
                        ],
                        "validation_precision": row_metrics[
                            "precision"
                        ],
                        "validation_recall": row_metrics[
                            "recall"
                        ],
                        "validation_f1": row_metrics["f1"],
                        "validation_event_recall": event_recall,
                        "validation_false_alert_episodes": (
                            false_alert_episodes
                        ),
                    }
                )
            except ImportError as error:
                comparison_records.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "status": "missing_dependency",
                        "error": str(error),
                    }
                )
            except (ArithmeticError, RuntimeError, TypeError, ValueError) as error:
                comparison_records.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "status": "failed",
                        "error": str(error),
                    }
                )

    comparison = pd.DataFrame(comparison_records)
    comparison.to_csv(
        output_directory / "model_feature_set_comparison.csv",
        index=False,
    )

    if not candidates:
        raise RuntimeError(
            "No model candidate completed successfully. "
            "Inspect model_feature_set_comparison.csv."
        )

    selected = max(
        candidates,
        key=lambda candidate: _candidate_selection_key(
            candidate,
            minimum_event_recall=minimum_event_recall,
        ),
    )

    validation_predictions = validation_rows[
        [
            TIMESTAMP_COLUMN,
            HIVE_COLUMN,
            SPLIT_COLUMN,
            target_column,
        ]
    ].copy()
    validation_predictions["raw_probability"] = (
        selected.validation_probabilities
    )
    validation_predictions["predicted_alert"] = (
        validation_predictions["raw_probability"]
        .ge(selected.threshold)
        .astype("int8")
    )

    test_probabilities = _positive_probabilities(
        selected.estimator,
        test_rows[selected.feature_columns],
    )
    test_predictions = test_rows[
        [
            TIMESTAMP_COLUMN,
            HIVE_COLUMN,
            SPLIT_COLUMN,
            target_column,
        ]
    ].copy()
    test_predictions["raw_probability"] = test_probabilities
    test_predictions["predicted_alert"] = (
        test_predictions["raw_probability"]
        .ge(selected.threshold)
        .astype("int8")
    )

    test_row_metrics = calculate_row_metrics(
        test_rows[target_column],
        test_probabilities,
        threshold=selected.threshold,
    )
    test_event_detection = build_event_detection_table(
        test_predictions,
        session_events,
        split="test",
        probability_column="raw_probability",
        threshold=selected.threshold,
        horizon_hours=horizon_hours,
    )
    test_event_recall = (
        float(test_event_detection["detected"].mean())
        if not test_event_detection.empty
        else 0.0
    )
    test_false_alert_episodes = count_false_alert_episodes(
        test_predictions,
        target_column=target_column,
        probability_column="raw_probability",
        threshold=selected.threshold,
        gap_hours=false_alert_gap_hours,
    )

    grouped_robustness = _run_grouped_hive_robustness(
        selected=selected,
        training_rows=train_rows,
        training_events=session_events.loc[
            session_events[SPLIT_COLUMN].eq("train")
        ],
        model_settings=settings["models"][
            selected.model_name
        ],
        target_column=target_column,
        maximum_negative_to_positive_ratio=negative_ratio,
        random_state=random_state,
        threshold=selected.threshold,
        horizon_hours=horizon_hours,
        false_alert_gap_hours=false_alert_gap_hours,
    )

    feature_importance = _extract_feature_importance(
        selected.estimator,
        selected.feature_columns,
    )

    session_summary = (
        session_events.groupby(
            SESSION_ID_COLUMN,
            observed=True,
        )
        .agg(
            session_start=("event_start", "min"),
            session_end=("event_start", "max"),
            hive_event_count=(EVENT_ID_COLUMN, "count"),
            unique_hives=(HIVE_COLUMN, "nunique"),
            splits=(SPLIT_COLUMN, lambda values: "|".join(
                sorted(set(values.astype(str)))
            )),
        )
        .reset_index()
    )

    validation_predictions.to_parquet(
        output_directory / "selected_validation_predictions.parquet",
        index=False,
    )
    test_predictions.to_parquet(
        output_directory / "selected_test_predictions.parquet",
        index=False,
    )
    selected.threshold_sweep.to_csv(
        output_directory / "selected_threshold_sweep.csv",
        index=False,
    )
    selected.validation_event_detection.to_csv(
        output_directory
        / "selected_validation_event_detection.csv",
        index=False,
    )
    test_event_detection.to_csv(
        output_directory / "selected_test_event_detection.csv",
        index=False,
    )
    grouped_robustness.to_csv(
        output_directory / "selected_grouped_hive_robustness.csv",
        index=False,
    )
    feature_importance.to_csv(
        output_directory / "selected_feature_importance.csv",
        index=False,
    )
    session_events.to_csv(
        output_directory / "reviewed_events_with_sessions.csv",
        index=False,
    )
    session_summary.to_csv(
        output_directory / "harvest_session_summary.csv",
        index=False,
    )

    joblib.dump(
        selected.estimator,
        model_directory / "selected_model.joblib",
    )
    _write_json(
        model_directory / "selected_feature_columns.json",
        {
            "feature_set": selected.feature_set_name,
            "features": selected.feature_columns,
        },
    )

    grouped_summary = {
        "fold_count": len(grouped_robustness),
        "mean_pr_auc": (
            float(grouped_robustness["pr_auc"].mean())
            if not grouped_robustness.empty
            else None
        ),
        "mean_event_recall": (
            float(
                grouped_robustness["event_recall"].mean()
            )
            if not grouped_robustness.empty
            else None
        ),
        "minimum_event_recall": (
            float(
                grouped_robustness["event_recall"].min()
            )
            if not grouped_robustness.empty
            else None
        ),
    }

    selected_metrics = {
        "research_stage": (
            "Model comparison completed. Probabilities are not yet "
            "calibrated and must not yet be presented as HUI."
        ),
        "selected_model": selected.model_name,
        "selected_feature_set": selected.feature_set_name,
        "selected_feature_count": len(
            selected.feature_columns
        ),
        "selected_threshold": selected.threshold,
        "selection_rule": (
            "Require the configured validation event recall when "
            "possible; then maximize validation PR-AUC, minimize "
            "false-alert episodes, prefer fewer features and prefer "
            "the simpler model in a tie."
        ),
        "validation": selected.validation_metrics,
        "test": {
            **test_row_metrics,
            "event_recall": test_event_recall,
            "false_alert_episodes": (
                test_false_alert_episodes
            ),
            "event_count": len(test_event_detection),
        },
        "grouped_hive_robustness": grouped_summary,
        "event_counts": {
            key: int(value)
            for key, value in session_events[
                SPLIT_COLUMN
            ].value_counts().to_dict().items()
        },
        "session_counts": {
            key: int(value)
            for key, value in (
                session_events.groupby(SPLIT_COLUMN)[
                    SESSION_ID_COLUMN
                ]
                .nunique()
                .to_dict()
                .items()
            )
        },
        "warnings": [
            (
                "The 12 reviewed hive-level events correspond to a "
                "smaller number of temporal harvest sessions."
            ),
            (
                "Validation contains two events and test contains one "
                "event; all performance claims must state these counts."
            ),
            (
                "The current output probabilities are uncalibrated "
                "model scores. HUI will be implemented only after "
                "training-only calibration."
            ),
            (
                "The target represents probable harvest activity within "
                "72 hours, not independently verified optimal honey "
                "maturity."
            ),
        ],
    }
    _write_json(
        output_directory / "selected_model_metrics.json",
        selected_metrics,
    )

    metadata = {
        "model_name": selected.model_name,
        "feature_set": selected.feature_set_name,
        "feature_columns": selected.feature_columns,
        "threshold": selected.threshold,
        "target_column": target_column,
        "horizon_hours": horizon_hours,
        "session_gap_hours": session_gap_hours,
        "training_probability_status": "uncalibrated",
        "next_required_stage": (
            "Training-only grouped out-of-fold probability calibration "
            "followed by HUI/HRSI/HRRoC implementation."
        ),
    }
    _write_json(
        model_directory / "model_metadata.json",
        metadata,
    )

    return {
        "selected_model": selected.model_name,
        "selected_feature_set": selected.feature_set_name,
        "selected_threshold": selected.threshold,
        "validation_pr_auc": selected.validation_metrics[
            "pr_auc"
        ],
        "validation_event_recall": selected.validation_metrics[
            "event_recall"
        ],
        "validation_false_alert_episodes": (
            selected.validation_metrics[
                "false_alert_episodes"
            ]
        ),
        "test_pr_auc": test_row_metrics["pr_auc"],
        "test_event_recall": test_event_recall,
        "test_event_count": len(test_event_detection),
        "hive_level_event_count": len(session_events),
        "temporal_session_count": int(
            session_events[SESSION_ID_COLUMN].nunique()
        ),
        "comparison_path": str(
            output_directory
            / "model_feature_set_comparison.csv"
        ),
        "metrics_path": str(
            output_directory / "selected_model_metrics.json"
        ),
        "model_path": str(
            model_directory / "selected_model.joblib"
        ),
    }
