from pathlib import Path
from runpy import run_path

import pandas as pd

_REVIEWER = run_path(
    str(
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "review_harvest_policy_generalization.py"
    )
)
evaluate_generalization = _REVIEWER[
    "evaluate_generalization"
]


def _detection(
    *,
    detected: bool,
    alert_rows: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "harvest_event_id": ["event_001"],
            "detected": [detected],
            "alert_rows": [alert_rows],
            "available_prediction_rows": [72],
        }
    )


def test_test_event_miss_blocks_generalization() -> None:
    result = evaluate_generalization(
        gate_summary={
            "ready_for_calibration": True,
        },
        validation_detection=_detection(
            detected=True,
            alert_rows=10,
        ),
        test_detection=_detection(
            detected=False,
            alert_rows=0,
        ),
        policy_metadata={
            "smoothing_window_hours": 12,
            "minimum_consecutive_hours": 4,
            "threshold": 0.01,
        },
    )

    assert result["validation_gate_passed"]
    assert not result["unchanged_test_event_supported"]
    assert not result["generalization_supported"]
    assert not result["calibration_allowed"]


def test_all_test_events_detected_is_limited_support() -> None:
    result = evaluate_generalization(
        gate_summary={
            "ready_for_calibration": True,
        },
        validation_detection=_detection(
            detected=True,
            alert_rows=10,
        ),
        test_detection=_detection(
            detected=True,
            alert_rows=5,
        ),
        policy_metadata={
            "smoothing_window_hours": 12,
            "minimum_consecutive_hours": 4,
            "threshold": 0.01,
        },
    )

    assert result["generalization_supported"]
    assert result["status"] == "limited_test_support"
    assert not result["calibration_allowed"]
