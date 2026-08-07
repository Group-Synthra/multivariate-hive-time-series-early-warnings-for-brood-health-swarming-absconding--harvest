from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

HIVE_COLUMN = "hive_id"
TIMESTAMP_COLUMN = "timestamp"
SPLIT_COLUMN = "split"
EVENT_ID_COLUMN = "harvest_event_id"
SESSION_ID_COLUMN = "harvest_session_id"


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_safe),
        encoding="utf-8",
    )


def add_contiguous_segment_id(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Restart temporal calculations after gaps and between hives."""
    _require_columns(
        predictions,
        {HIVE_COLUMN, TIMESTAMP_COLUMN},
        frame_name="Predictions",
    )

    frame = predictions.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        frame[TIMESTAMP_COLUMN],
        errors="raise",
    )
    frame = frame.sort_values(
        [HIVE_COLUMN, TIMESTAMP_COLUMN]
    ).reset_index(drop=True)

    duplicate_count = int(
        frame.duplicated([HIVE_COLUMN, TIMESTAMP_COLUMN]).sum()
    )
    if duplicate_count:
        raise ValueError(
            "Prediction rows contain duplicate hive-timestamp keys: "
            f"{duplicate_count}"
        )

    previous_hive = frame[HIVE_COLUMN].shift()
    elapsed_hours = (
        frame[TIMESTAMP_COLUMN]
        .sub(frame[TIMESTAMP_COLUMN].shift())
        .dt.total_seconds()
        .div(3600)
    )
    starts_segment = (
        frame[HIVE_COLUMN].ne(previous_hive)
        | elapsed_hours.ne(1.0)
    )
    frame["_segment_id"] = starts_segment.cumsum().astype("int64")
    return frame


def add_smoothed_probability(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
    smoothing_window_hours: int,
) -> pd.DataFrame:
    if smoothing_window_hours <= 0:
        raise ValueError(
            "smoothing_window_hours must be greater than zero"
        )

    _require_columns(
        predictions,
        {
            HIVE_COLUMN,
            "_segment_id",
            probability_column,
        },
        frame_name="Segmented predictions",
    )

    frame = predictions.copy()
    frame["smoothed_probability"] = (
        frame.groupby(
            [HIVE_COLUMN, "_segment_id"],
            sort=False,
        )[probability_column]
        .transform(
            lambda values: values.rolling(
                window=smoothing_window_hours,
                min_periods=smoothing_window_hours,
            ).mean()
        )
    )
    return frame


def add_consecutive_run_length(
    predictions: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")

    _require_columns(
        predictions,
        {
            HIVE_COLUMN,
            "_segment_id",
            "smoothed_probability",
        },
        frame_name="Smoothed predictions",
    )

    frame = predictions.copy()
    above = frame["smoothed_probability"].ge(threshold).fillna(False)
    segment_keys = [frame[HIVE_COLUMN], frame["_segment_id"]]
    run_group = (~above).groupby(segment_keys, sort=False).cumsum()
    frame["_consecutive_above_threshold"] = (
        above.astype("int64")
        .groupby(
            [frame[HIVE_COLUMN], frame["_segment_id"], run_group],
            sort=False,
        )
        .cumsum()
    )
    return frame


def apply_alert_policy(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
    smoothing_window_hours: int,
    threshold: float,
    minimum_consecutive_hours: int,
) -> pd.DataFrame:
    if minimum_consecutive_hours <= 0:
        raise ValueError(
            "minimum_consecutive_hours must be greater than zero"
        )

    frame = add_contiguous_segment_id(predictions)
    frame = add_smoothed_probability(
        frame,
        probability_column=probability_column,
        smoothing_window_hours=smoothing_window_hours,
    )
    frame = add_consecutive_run_length(frame, threshold=threshold)
    frame["alert"] = (
        frame["_consecutive_above_threshold"]
        .ge(minimum_consecutive_hours)
        .astype("int8")
    )
    return frame


def calculate_alert_metrics(
    predictions: pd.DataFrame,
    *,
    target_column: str,
    alert_column: str = "alert",
) -> dict[str, Any]:
    _require_columns(
        predictions,
        {target_column, alert_column},
        frame_name="Alert predictions",
    )

    target = predictions[target_column].astype("int8")
    alert = predictions[alert_column].astype("int8")

    true_positive = int(target.eq(1).mul(alert.eq(1)).sum())
    false_positive = int(target.eq(0).mul(alert.eq(1)).sum())
    false_negative = int(target.eq(1).mul(alert.eq(0)).sum())
    true_negative = int(target.eq(0).mul(alert.eq(0)).sum())

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else 0.0
    )
    recall = (
        true_positive / recall_denominator
        if recall_denominator
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_negatives": true_negative,
        "alert_rows": int(alert.sum()),
    }


def count_false_alert_episodes(
    predictions: pd.DataFrame,
    *,
    target_column: str,
    alert_column: str = "alert",
    gap_hours: int,
) -> int:
    if gap_hours < 0:
        raise ValueError("gap_hours cannot be negative")

    false_alerts = predictions.loc[
        predictions[target_column].eq(0)
        & predictions[alert_column].eq(1)
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
    horizon_hours: int,
    alert_column: str = "alert",
) -> pd.DataFrame:
    _require_columns(
        events,
        {
            HIVE_COLUMN,
            EVENT_ID_COLUMN,
            SESSION_ID_COLUMN,
            "event_start",
            SPLIT_COLUMN,
        },
        frame_name="Reviewed events",
    )

    split_events = events.loc[events[SPLIT_COLUMN].eq(split)].copy()
    split_events["event_start"] = pd.to_datetime(
        split_events["event_start"],
        errors="raise",
    )
    split_events = split_events.sort_values("event_start")
    horizon = pd.Timedelta(hours=horizon_hours)
    records: list[dict[str, Any]] = []

    for event in split_events.itertuples(index=False):
        event_rows = predictions.loc[
            predictions[HIVE_COLUMN].eq(event.hive_id)
            & predictions[TIMESTAMP_COLUMN].ge(
                event.event_start - horizon
            )
            & predictions[TIMESTAMP_COLUMN].lt(event.event_start)
        ].sort_values(TIMESTAMP_COLUMN)

        alert_rows = event_rows.loc[event_rows[alert_column].eq(1)]
        detected = not alert_rows.empty
        first_alert_time = (
            alert_rows[TIMESTAMP_COLUMN].iloc[0]
            if detected
            else pd.NaT
        )
        lead_hours = (
            (event.event_start - first_alert_time).total_seconds()
            / 3600
            if detected
            else np.nan
        )

        record = {
            EVENT_ID_COLUMN: event.harvest_event_id,
            SESSION_ID_COLUMN: event.harvest_session_id,
            HIVE_COLUMN: event.hive_id,
            SPLIT_COLUMN: split,
            "event_start": event.event_start,
            "available_prediction_rows": len(event_rows),
            "alert_rows": len(alert_rows),
            "detected": bool(detected),
            "first_alert_time": first_alert_time,
            "lead_hours": lead_hours,
        }
        for column in (
            "raw_probability",
            "smoothed_probability",
        ):
            record[f"maximum_{column}"] = (
                float(event_rows[column].max())
                if column in event_rows and not event_rows.empty
                else np.nan
            )
        records.append(record)

    return pd.DataFrame(records)


def summarize_event_detection(
    detection: pd.DataFrame,
) -> dict[str, Any]:
    if detection.empty:
        return {
            "event_count": 0,
            "detected_event_count": 0,
            "event_recall": 0.0,
            "session_count": 0,
            "detected_session_count": 0,
            "session_recall": 0.0,
            "median_lead_hours": None,
            "minimum_lead_hours": None,
        }

    detected = detection["detected"].astype(bool)
    detected_rows = detection.loc[detected]
    session_detected = (
        detection.groupby(
            SESSION_ID_COLUMN,
            observed=True,
        )["detected"]
        .max()
    )
    return {
        "event_count": len(detection),
        "detected_event_count": int(detected.sum()),
        "event_recall": float(detected.mean()),
        "session_count": int(
            detection[SESSION_ID_COLUMN].nunique()
        ),
        "detected_session_count": int(session_detected.sum()),
        "session_recall": float(session_detected.mean()),
        "median_lead_hours": (
            float(detected_rows["lead_hours"].median())
            if not detected_rows.empty
            else None
        ),
        "minimum_lead_hours": (
            float(detected_rows["lead_hours"].min())
            if not detected_rows.empty
            else None
        ),
    }


def evaluate_policy(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    split: str,
    target_column: str,
    false_alert_gap_hours: int,
    horizon_hours: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    row_metrics = calculate_alert_metrics(
        predictions,
        target_column=target_column,
    )
    detection = build_event_detection_table(
        predictions,
        events,
        split=split,
        horizon_hours=horizon_hours,
    )
    event_metrics = summarize_event_detection(detection)
    false_alert_episodes = count_false_alert_episodes(
        predictions,
        target_column=target_column,
        gap_hours=false_alert_gap_hours,
    )
    negative_rows = int(predictions[target_column].eq(0).sum())
    false_alerts_per_30_hive_days = (
        false_alert_episodes * 720 / negative_rows
        if negative_rows
        else 0.0
    )
    return (
        {
            **row_metrics,
            **event_metrics,
            "false_alert_episodes": false_alert_episodes,
            "false_alerts_per_30_hive_days": (
                false_alerts_per_30_hive_days
            ),
        },
        detection,
    )


def _build_threshold_grid(
    probabilities: pd.Series,
    *,
    threshold_grid_points: int,
) -> np.ndarray:
    if threshold_grid_points < 3:
        raise ValueError(
            "threshold_grid_points must be at least three"
        )

    finite = probabilities.dropna().to_numpy(dtype=float)
    if finite.size == 0:
        raise ValueError(
            "No finite smoothed probabilities are available."
        )

    quantiles = np.quantile(
        finite,
        np.linspace(0.0, 1.0, threshold_grid_points),
    )
    linear = np.linspace(
        max(0.001, float(finite.min())),
        min(0.999, float(finite.max())),
        threshold_grid_points,
    )
    return np.unique(
        np.clip(
            np.concatenate([quantiles, linear]),
            0.0,
            1.0,
        )
    )


def select_alert_policy(
    validation_predictions: pd.DataFrame,
    validation_events: pd.DataFrame,
    *,
    target_column: str,
    probability_column: str,
    smoothing_windows_hours: list[int],
    minimum_consecutive_hours: list[int],
    threshold_grid_points: int,
    minimum_validation_event_recall: float,
    minimum_median_lead_hours: float,
    false_alert_gap_hours: int,
    horizon_hours: int,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    segmented = add_contiguous_segment_id(validation_predictions)
    records: list[dict[str, Any]] = []
    evaluated: dict[
        tuple[int, int, float],
        tuple[pd.DataFrame, pd.DataFrame],
    ] = {}

    for smoothing_window in smoothing_windows_hours:
        smoothed = add_smoothed_probability(
            segmented,
            probability_column=probability_column,
            smoothing_window_hours=smoothing_window,
        )
        thresholds = _build_threshold_grid(
            smoothed["smoothed_probability"],
            threshold_grid_points=threshold_grid_points,
        )

        for threshold in thresholds:
            runs = add_consecutive_run_length(
                smoothed,
                threshold=float(threshold),
            )
            for consecutive_hours in minimum_consecutive_hours:
                policy_predictions = runs.copy()
                policy_predictions["alert"] = (
                    policy_predictions[
                        "_consecutive_above_threshold"
                    ]
                    .ge(consecutive_hours)
                    .astype("int8")
                )
                metrics, detection = evaluate_policy(
                    policy_predictions,
                    validation_events,
                    split="validation",
                    target_column=target_column,
                    false_alert_gap_hours=false_alert_gap_hours,
                    horizon_hours=horizon_hours,
                )
                key = (
                    int(smoothing_window),
                    int(consecutive_hours),
                    float(threshold),
                )
                evaluated[key] = (
                    policy_predictions,
                    detection,
                )
                records.append(
                    {
                        "smoothing_window_hours": int(
                            smoothing_window
                        ),
                        "minimum_consecutive_hours": int(
                            consecutive_hours
                        ),
                        "threshold": float(threshold),
                        **metrics,
                    }
                )

    sweep = pd.DataFrame(records)
    eligible = sweep.loc[
        sweep["event_recall"].ge(
            minimum_validation_event_recall
        )
        & sweep["median_lead_hours"]
        .fillna(-np.inf)
        .ge(minimum_median_lead_hours)
    ].copy()

    if eligible.empty:
        ranked = sweep.sort_values(
            [
                "event_recall",
                "median_lead_hours",
                "false_alert_episodes",
                "precision",
                "f1",
                "smoothing_window_hours",
                "minimum_consecutive_hours",
            ],
            ascending=[False, False, True, False, False, True, True],
        )
        selection_status = (
            "fallback_no_policy_met_all_constraints"
        )
    else:
        ranked = eligible.sort_values(
            [
                "false_alert_episodes",
                "precision",
                "f1",
                "median_lead_hours",
                "smoothing_window_hours",
                "minimum_consecutive_hours",
                "threshold",
            ],
            ascending=[True, False, False, False, True, True, False],
        )
        selection_status = "constraints_satisfied"

    selected_row = ranked.iloc[0].to_dict()
    selected_key = (
        int(selected_row["smoothing_window_hours"]),
        int(selected_row["minimum_consecutive_hours"]),
        float(selected_row["threshold"]),
    )
    selected_predictions, selected_detection = evaluated[selected_key]
    selected_policy = {
        "selection_status": selection_status,
        "smoothing_window_hours": selected_key[0],
        "minimum_consecutive_hours": selected_key[1],
        "threshold": selected_key[2],
    }
    return (
        selected_policy,
        sweep.sort_values(
            ["event_recall", "false_alert_episodes", "precision"],
            ascending=[False, True, False],
        ).reset_index(drop=True),
        selected_predictions,
        selected_detection,
    )


def evaluate_existing_policy(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    split: str,
    target_column: str,
    false_alert_gap_hours: int,
    horizon_hours: int,
) -> dict[str, Any]:
    if "predicted_alert" not in predictions:
        return {}

    baseline = predictions.copy()
    baseline["alert"] = baseline["predicted_alert"].astype("int8")
    baseline[TIMESTAMP_COLUMN] = pd.to_datetime(
        baseline[TIMESTAMP_COLUMN],
        errors="raise",
    )
    metrics, _ = evaluate_policy(
        baseline,
        events,
        split=split,
        target_column=target_column,
        false_alert_gap_hours=false_alert_gap_hours,
        horizon_hours=horizon_hours,
    )
    return metrics


def run_alert_policy_refinement_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = config["alert_policy_refinement"]

    validation_path = _resolve_path(
        root,
        settings["validation_predictions_path"],
    )
    test_path = _resolve_path(
        root,
        settings["test_predictions_path"],
    )
    events_path = _resolve_path(
        root,
        settings["reviewed_events_path"],
    )
    output_directory = _resolve_path(
        root,
        settings["output_directory"],
    )
    deployment_metadata_path = _resolve_path(
        root,
        settings["deployment_metadata_path"],
    )

    validation = pd.read_parquet(validation_path)
    test = pd.read_parquet(test_path)
    events = pd.read_csv(events_path)
    events["event_start"] = pd.to_datetime(
        events["event_start"],
        errors="raise",
    )

    target_column = str(settings["target_column"])
    probability_column = str(settings["probability_column"])
    horizon_hours = int(settings["horizon_hours"])
    false_alert_gap_hours = int(
        settings["false_alert_gap_hours"]
    )

    required_columns = {
        TIMESTAMP_COLUMN,
        HIVE_COLUMN,
        SPLIT_COLUMN,
        target_column,
        probability_column,
    }
    _require_columns(
        validation,
        required_columns,
        frame_name="Validation predictions",
    )
    _require_columns(
        test,
        required_columns,
        frame_name="Test predictions",
    )
    validation[TIMESTAMP_COLUMN] = pd.to_datetime(
        validation[TIMESTAMP_COLUMN],
        errors="raise",
    )
    test[TIMESTAMP_COLUMN] = pd.to_datetime(
        test[TIMESTAMP_COLUMN],
        errors="raise",
    )

    baseline_validation = evaluate_existing_policy(
        validation,
        events,
        split="validation",
        target_column=target_column,
        false_alert_gap_hours=false_alert_gap_hours,
        horizon_hours=horizon_hours,
    )

    (
        selected_policy,
        sweep,
        selected_validation_predictions,
        selected_validation_detection,
    ) = select_alert_policy(
        validation,
        events,
        target_column=target_column,
        probability_column=probability_column,
        smoothing_windows_hours=[
            int(value)
            for value in settings["smoothing_windows_hours"]
        ],
        minimum_consecutive_hours=[
            int(value)
            for value in settings["minimum_consecutive_hours"]
        ],
        threshold_grid_points=int(settings["threshold_grid_points"]),
        minimum_validation_event_recall=float(
            settings["minimum_validation_event_recall"]
        ),
        minimum_median_lead_hours=float(
            settings["minimum_median_lead_hours"]
        ),
        false_alert_gap_hours=false_alert_gap_hours,
        horizon_hours=horizon_hours,
    )

    validation_metrics, _ = evaluate_policy(
        selected_validation_predictions,
        events,
        split="validation",
        target_column=target_column,
        false_alert_gap_hours=false_alert_gap_hours,
        horizon_hours=horizon_hours,
    )

    selected_test_predictions = apply_alert_policy(
        test,
        probability_column=probability_column,
        smoothing_window_hours=int(
            selected_policy["smoothing_window_hours"]
        ),
        threshold=float(selected_policy["threshold"]),
        minimum_consecutive_hours=int(
            selected_policy["minimum_consecutive_hours"]
        ),
    )
    test_metrics, test_detection = evaluate_policy(
        selected_test_predictions,
        events,
        split="test",
        target_column=target_column,
        false_alert_gap_hours=false_alert_gap_hours,
        horizon_hours=horizon_hours,
    )

    prevalence = float(validation[target_column].mean())
    precision_lift = (
        validation_metrics["precision"] / prevalence
        if prevalence
        else 0.0
    )
    baseline_false_alerts = baseline_validation.get(
        "false_alert_episodes"
    )
    selected_false_alerts = validation_metrics[
        "false_alert_episodes"
    ]
    false_alert_reduction_fraction = (
        (baseline_false_alerts - selected_false_alerts)
        / baseline_false_alerts
        if baseline_false_alerts
        else None
    )

    gate = settings["readiness_gate"]
    minimum_reduction = float(
        gate["minimum_false_alert_reduction_fraction"]
    )
    minimum_precision_lift = float(
        gate["minimum_precision_lift_over_prevalence"]
    )
    constraints_satisfied = (
        selected_policy["selection_status"]
        == "constraints_satisfied"
    )
    reduction_passed = bool(
        false_alert_reduction_fraction is not None
        and false_alert_reduction_fraction >= minimum_reduction
    )
    precision_passed = precision_lift >= minimum_precision_lift
    ready_for_calibration_review = bool(
        constraints_satisfied
        and reduction_passed
        and precision_passed
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    deployment_metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sweep.to_csv(
        output_directory / "alert_policy_sweep.csv",
        index=False,
    )
    sweep.head(50).to_csv(
        output_directory / "top_alert_policies.csv",
        index=False,
    )

    validation_output_columns = [
        column
        for column in selected_validation_predictions.columns
        if not column.startswith("_")
    ]
    selected_validation_predictions[
        validation_output_columns
    ].to_parquet(
        output_directory
        / "selected_validation_alert_predictions.parquet",
        index=False,
    )
    test_output_columns = [
        column
        for column in selected_test_predictions.columns
        if not column.startswith("_")
    ]
    selected_test_predictions[test_output_columns].to_parquet(
        output_directory
        / "selected_test_alert_predictions.parquet",
        index=False,
    )
    selected_validation_detection.to_csv(
        output_directory
        / "selected_validation_event_detection.csv",
        index=False,
    )
    test_detection.to_csv(
        output_directory / "selected_test_event_detection.csv",
        index=False,
    )

    summary = {
        "selected_policy": selected_policy,
        "baseline_validation_policy": baseline_validation,
        "selected_validation_policy": validation_metrics,
        "unchanged_test_evaluation": test_metrics,
        "validation_prevalence": prevalence,
        "validation_precision_lift_over_prevalence": precision_lift,
        "validation_false_alert_reduction_fraction": (
            false_alert_reduction_fraction
        ),
        "readiness_gate": {
            "minimum_false_alert_reduction_fraction": (
                minimum_reduction
            ),
            "minimum_precision_lift_over_prevalence": (
                minimum_precision_lift
            ),
            "constraints_satisfied": constraints_satisfied,
            "false_alert_reduction_passed": reduction_passed,
            "precision_lift_passed": precision_passed,
            "ready_for_calibration_review": (
                ready_for_calibration_review
            ),
        },
        "interpretation": (
            "The policy consolidates hourly model scores into alert "
            "episodes. Test metrics are a one-event case study and "
            "are not used to tune the policy."
        ),
        "warnings": [
            (
                "The policy was selected using only two reviewed "
                "validation events."
            ),
            (
                "The model score is not probability calibrated and "
                "must not yet be displayed as HUI."
            ),
            (
                "The target represents probable harvest activity "
                "within 72 hours, not verified optimal maturity."
            ),
        ],
    }
    _write_json(
        output_directory / "selected_alert_policy.json",
        summary,
    )
    _write_json(
        deployment_metadata_path,
        {
            **selected_policy,
            "probability_column": probability_column,
            "horizon_hours": horizon_hours,
            "false_alert_gap_hours": false_alert_gap_hours,
            "calibration_status": "not_calibrated",
            "ready_for_calibration_review": (
                ready_for_calibration_review
            ),
        },
    )

    return {
        "selected_policy": selected_policy,
        "validation_event_recall": validation_metrics[
            "event_recall"
        ],
        "validation_median_lead_hours": validation_metrics[
            "median_lead_hours"
        ],
        "validation_precision": validation_metrics["precision"],
        "validation_false_alert_episodes": selected_false_alerts,
        "baseline_false_alert_episodes": baseline_false_alerts,
        "false_alert_reduction_fraction": (
            false_alert_reduction_fraction
        ),
        "test_event_recall": test_metrics["event_recall"],
        "test_false_alert_episodes": test_metrics[
            "false_alert_episodes"
        ],
        "ready_for_calibration_review": (
            ready_for_calibration_review
        ),
        "summary_path": str(
            output_directory / "selected_alert_policy.json"
        ),
    }
