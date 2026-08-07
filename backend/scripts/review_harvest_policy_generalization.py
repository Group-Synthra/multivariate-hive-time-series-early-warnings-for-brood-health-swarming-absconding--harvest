from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )
    return normalized.isin(
        {"true", "1", "yes", "y"}
    )


def _summarize_detection(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    required = {
        "harvest_event_id",
        "detected",
        "alert_rows",
        "available_prediction_rows",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Event-detection table is missing columns: "
            f"{missing}"
        )

    event_count = len(frame)
    detected = _as_boolean(frame["detected"])
    detected_count = int(detected.sum())
    event_recall = (
        detected_count / event_count
        if event_count > 0
        else None
    )
    total_alert_rows = int(
        frame["alert_rows"].fillna(0).sum()
    )
    total_available_rows = int(
        frame["available_prediction_rows"].fillna(0).sum()
    )
    alert_fraction_in_event_windows = (
        total_alert_rows / total_available_rows
        if total_available_rows > 0
        else None
    )

    missed_events = (
        frame.loc[
            ~detected,
            "harvest_event_id",
        ]
        .astype(str)
        .tolist()
    )

    return {
        "event_count": event_count,
        "detected_event_count": detected_count,
        "event_recall": event_recall,
        "total_alert_rows_in_event_windows": total_alert_rows,
        "total_available_prediction_rows": (
            total_available_rows
        ),
        "alert_fraction_in_event_windows": (
            alert_fraction_in_event_windows
        ),
        "missed_event_ids": missed_events,
    }


def evaluate_generalization(
    *,
    gate_summary: dict[str, Any],
    validation_detection: pd.DataFrame,
    test_detection: pd.DataFrame,
    policy_metadata: dict[str, Any],
) -> dict[str, Any]:
    validation = _summarize_detection(
        validation_detection
    )
    test = _summarize_detection(test_detection)

    validation_gate_passed = (
        gate_summary.get("ready_for_calibration") is True
    )
    test_event_supported = (
        test["event_count"] > 0
        and test["detected_event_count"]
        == test["event_count"]
    )

    generalization_supported = bool(
        validation_gate_passed and test_event_supported
    )

    if not validation_gate_passed:
        status = "validation_policy_not_eligible"
        decision = (
            "Do not calibrate. The validation policy did not "
            "satisfy the operational research gate."
        )
    elif not test_event_supported:
        status = "validation_eligible_test_event_missed"
        decision = (
            "Do not calibrate or deploy. The validation-selected "
            "policy missed at least one unchanged held-out test "
            "event."
        )
    else:
        status = "limited_test_support"
        decision = (
            "The unchanged policy detected all held-out test events, "
            "but evidence remains provisional because only one test "
            "event is available."
        )

    result = {
        "status": status,
        "validation_gate_passed": validation_gate_passed,
        "unchanged_test_event_supported": (
            test_event_supported
        ),
        "generalization_supported": (
            generalization_supported
        ),
        "calibration_allowed": False,
        "deployment_allowed": False,
        "policy": {
            "smoothing_window_hours": policy_metadata.get(
                "smoothing_window_hours"
            ),
            "minimum_consecutive_hours": policy_metadata.get(
                "minimum_consecutive_hours"
            ),
            "threshold": policy_metadata.get("threshold"),
        },
        "validation": validation,
        "test": test,
        "decision": decision,
        "warnings": [
            (
                "The policy was selected from only two reviewed "
                "validation events."
            ),
            (
                "The unchanged held-out evaluation contains only "
                "one reviewed test event."
            ),
            (
                "The target represents probable harvest activity "
                "within 72 hours, not verified honey maturity."
            ),
            (
                "Raw classifier scores are not calibrated "
                "probabilities and must not be displayed as HUI."
            ),
        ],
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review generalization of the validation-selected "
            "harvest alert policy on the unchanged held-out test "
            "event."
        )
    )
    parser.add_argument(
        "--report-root",
        default=(
            "artifacts/reports/harvesting/reviewed"
        ),
    )
    parser.add_argument(
        "--model-root",
        default=(
            "artifacts/models/harvesting/research_v2"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    backend_root = Path(__file__).resolve().parents[1]
    report_root = backend_root / arguments.report_root
    model_root = backend_root / arguments.model_root

    gate_summary_path = (
        report_root
        / "alert_policy_gate"
        / "research_gate_summary.json"
    )
    validation_detection_path = (
        report_root
        / "alert_policy_gate"
        / "research_safe_validation_event_detection.csv"
    )
    test_detection_path = (
        report_root
        / "alert_policy_gate"
        / "research_safe_test_event_detection.csv"
    )
    policy_metadata_path = (
        model_root / "research_gate_policy.json"
    )

    gate_summary = _read_json(gate_summary_path)
    validation_detection = pd.read_csv(
        validation_detection_path
    )
    test_detection = pd.read_csv(
        test_detection_path
    )
    policy_metadata = _read_json(
        policy_metadata_path
    )

    result = evaluate_generalization(
        gate_summary=gate_summary,
        validation_detection=validation_detection,
        test_detection=test_detection,
        policy_metadata=policy_metadata,
    )

    output_path = (
        report_root
        / "alert_policy_gate"
        / "classification_generalization_review.json"
    )
    _write_json(output_path, result)

    print(json.dumps(result, indent=2))
    print("\nCreated:", output_path)


if __name__ == "__main__":
    main()
