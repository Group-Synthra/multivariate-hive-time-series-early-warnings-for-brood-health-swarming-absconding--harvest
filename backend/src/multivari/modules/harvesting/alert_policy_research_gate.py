from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from multivari.modules.harvesting.alert_policy_refinement import (
    apply_alert_policy,
    evaluate_policy,
)

TIMESTAMP_COLUMN = "timestamp"


def _resolve_path(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_safe),
        encoding="utf-8",
    )


def add_operational_metrics(
    sweep: pd.DataFrame,
    *,
    prevalence: float,
) -> pd.DataFrame:
    required = {
        "precision",
        "false_positives",
        "true_negatives",
        "alert_rows",
        "true_positives",
        "false_negatives",
    }
    missing = sorted(required.difference(sweep.columns))
    if missing:
        raise ValueError(
            "Alert-policy sweep is missing required columns: "
            f"{missing}"
        )

    result = sweep.copy()
    total_rows = (
        result["true_positives"]
        + result["false_positives"]
        + result["false_negatives"]
        + result["true_negatives"]
    )
    negative_rows = (
        result["false_positives"] + result["true_negatives"]
    )

    result["alert_fraction"] = (
        result["alert_rows"] / total_rows.replace(0, np.nan)
    )
    result["false_positive_rate"] = (
        result["false_positives"]
        / negative_rows.replace(0, np.nan)
    )
    result["precision_lift_over_prevalence"] = (
        result["precision"] / prevalence
        if prevalence > 0
        else np.nan
    )
    return result


def select_research_safe_policy(
    sweep: pd.DataFrame,
    *,
    prevalence: float,
    baseline_metrics: dict[str, Any],
    minimum_event_recall: float,
    minimum_median_lead_hours: float,
    minimum_precision_lift: float,
    require_precision_at_least_baseline: bool,
    require_false_positive_rows_no_worse_than_baseline: bool,
    require_alert_fraction_no_worse_than_baseline: bool,
) -> tuple[dict[str, Any] | None, pd.DataFrame, dict[str, Any]]:
    enriched = add_operational_metrics(
        sweep,
        prevalence=prevalence,
    )

    baseline_total_rows = sum(
        int(baseline_metrics[key])
        for key in (
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
        )
    )
    baseline_alert_fraction = (
        float(baseline_metrics["alert_rows"])
        / baseline_total_rows
    )
    baseline_precision = float(baseline_metrics["precision"])
    baseline_false_positives = int(
        baseline_metrics["false_positives"]
    )

    eligibility = (
        enriched["event_recall"].ge(minimum_event_recall)
        & enriched["median_lead_hours"]
        .fillna(-np.inf)
        .ge(minimum_median_lead_hours)
        & enriched["precision_lift_over_prevalence"]
        .ge(minimum_precision_lift)
    )

    if require_precision_at_least_baseline:
        eligibility &= enriched["precision"].ge(
            baseline_precision
        )

    if require_false_positive_rows_no_worse_than_baseline:
        eligibility &= enriched["false_positives"].le(
            baseline_false_positives
        )

    if require_alert_fraction_no_worse_than_baseline:
        eligibility &= enriched["alert_fraction"].le(
            baseline_alert_fraction
        )

    eligible = enriched.loc[eligibility].copy()
    eligible = eligible.sort_values(
        [
            "f1",
            "precision",
            "false_positives",
            "alert_fraction",
            "false_alert_episodes",
            "median_lead_hours",
            "smoothing_window_hours",
            "minimum_consecutive_hours",
        ],
        ascending=[
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    gate_summary = {
        "validation_prevalence": prevalence,
        "baseline_precision": baseline_precision,
        "baseline_false_positives": baseline_false_positives,
        "baseline_alert_fraction": baseline_alert_fraction,
        "minimum_event_recall": minimum_event_recall,
        "minimum_median_lead_hours": (
            minimum_median_lead_hours
        ),
        "minimum_precision_lift_over_prevalence": (
            minimum_precision_lift
        ),
        "candidate_count": len(enriched),
        "eligible_candidate_count": len(eligible),
    }

    if eligible.empty:
        return None, enriched, gate_summary

    return eligible.iloc[0].to_dict(), enriched, gate_summary


def run_alert_policy_research_gate_from_config(
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
    settings = config["alert_policy_research_gate"]

    sweep_path = _resolve_path(
        root,
        settings["policy_sweep_path"],
    )
    summary_path = _resolve_path(
        root,
        settings["policy_summary_path"],
    )
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
    deployment_path = _resolve_path(
        root,
        settings["deployment_metadata_path"],
    )

    sweep = pd.read_csv(sweep_path)
    prior_summary = _read_json(summary_path)
    validation = pd.read_parquet(validation_path)
    test = pd.read_parquet(test_path)
    events = pd.read_csv(events_path)

    target_column = str(settings["target_column"])
    probability_column = str(
        settings["probability_column"]
    )
    horizon_hours = int(settings["horizon_hours"])
    false_alert_gap_hours = int(
        settings["false_alert_gap_hours"]
    )

    for frame in (validation, test):
        frame[TIMESTAMP_COLUMN] = pd.to_datetime(
            frame[TIMESTAMP_COLUMN],
            errors="raise",
        )
    events["event_start"] = pd.to_datetime(
        events["event_start"],
        errors="raise",
    )

    prevalence = float(validation[target_column].mean())
    baseline_metrics = prior_summary[
        "baseline_validation_policy"
    ]

    selected, enriched, gate_summary = (
        select_research_safe_policy(
            sweep,
            prevalence=prevalence,
            baseline_metrics=baseline_metrics,
            minimum_event_recall=float(
                settings[
                    "minimum_validation_event_recall"
                ]
            ),
            minimum_median_lead_hours=float(
                settings["minimum_median_lead_hours"]
            ),
            minimum_precision_lift=float(
                settings[
                    "minimum_precision_lift_over_prevalence"
                ]
            ),
            require_precision_at_least_baseline=bool(
                settings[
                    "require_precision_at_least_baseline"
                ]
            ),
            require_false_positive_rows_no_worse_than_baseline=bool(
                settings[
                    "require_false_positive_rows_no_worse_than_baseline"
                ]
            ),
            require_alert_fraction_no_worse_than_baseline=bool(
                settings[
                    "require_alert_fraction_no_worse_than_baseline"
                ]
            ),
        )
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    deployment_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    enriched.to_csv(
        output_directory
        / "policy_sweep_with_operational_metrics.csv",
        index=False,
    )

    if selected is None:
        closest = enriched.loc[
            enriched["event_recall"].ge(
                float(
                    settings[
                        "minimum_validation_event_recall"
                    ]
                )
            )
        ].sort_values(
            [
                "f1",
                "precision",
                "false_positives",
            ],
            ascending=[False, False, True],
        ).head(100)
        closest.to_csv(
            output_directory
            / "closest_non_deployable_policies.csv",
            index=False,
        )

        result = {
            "status": "no_research_safe_policy",
            "ready_for_calibration": False,
            "gate": gate_summary,
            "reason": (
                "No temporal policy simultaneously preserved event "
                "detection, useful lead time, baseline precision, "
                "baseline false-positive burden and baseline alert "
                "occupancy."
            ),
            "recommended_next_stage": (
                "Freeze the event classifier as a benchmark and run "
                "future hive-weight forecasting."
            ),
        }
        _write_json(
            output_directory / "research_gate_summary.json",
            result,
        )
        _write_json(
            deployment_path,
            {
                "deployment_allowed": False,
                "reason": result["reason"],
            },
        )
        return result

    policy = {
        "smoothing_window_hours": int(
            selected["smoothing_window_hours"]
        ),
        "minimum_consecutive_hours": int(
            selected["minimum_consecutive_hours"]
        ),
        "threshold": float(selected["threshold"]),
    }

    validation_policy = apply_alert_policy(
        validation,
        probability_column=probability_column,
        **policy,
    )
    validation_metrics, validation_detection = (
        evaluate_policy(
            validation_policy,
            events,
            split="validation",
            target_column=target_column,
            false_alert_gap_hours=(
                false_alert_gap_hours
            ),
            horizon_hours=horizon_hours,
        )
    )

    test_policy = apply_alert_policy(
        test,
        probability_column=probability_column,
        **policy,
    )
    test_metrics, test_detection = evaluate_policy(
        test_policy,
        events,
        split="test",
        target_column=target_column,
        false_alert_gap_hours=false_alert_gap_hours,
        horizon_hours=horizon_hours,
    )

    validation_detection.to_csv(
        output_directory
        / "research_safe_validation_event_detection.csv",
        index=False,
    )
    test_detection.to_csv(
        output_directory
        / "research_safe_test_event_detection.csv",
        index=False,
    )

    result = {
        "status": "research_safe_policy_found",
        "ready_for_calibration": True,
        "policy": policy,
        "gate": gate_summary,
        "validation": validation_metrics,
        "test_case_study": test_metrics,
        "warning": (
            "Validation contains two events and test contains one "
            "event; any later calibration remains provisional."
        ),
    }
    _write_json(
        output_directory / "research_gate_summary.json",
        result,
    )
    _write_json(
        deployment_path,
        {
            "deployment_allowed": False,
            "calibration_required": True,
            **policy,
        },
    )
    return result
