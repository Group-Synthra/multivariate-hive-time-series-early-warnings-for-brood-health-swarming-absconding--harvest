from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "scripts/export_classifier_derived_hui_viva_dashboard.py"
)
SPEC = importlib.util.spec_from_file_location("hui_viva_export", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_hui_class_boundaries() -> None:
    assert MODULE._classify_hui(39.999) == "Not Ready"
    assert MODULE._classify_hui(40.0) == "Approaching Harvest"
    assert MODULE._classify_hui(60.0) == "Ready"
    assert MODULE._classify_hui(80.0) == "High-Priority Harvest"


def test_stability_is_high_for_constant_hui() -> None:
    frame = pd.DataFrame({"classifier_derived_hui": [50.0] * 24})
    assert MODULE._recent_hui_stability(frame) == 100.0


def test_rate_label_uses_fixed_slope_boundaries() -> None:
    assert MODULE._rate_label(0.6) == "Increasing"
    assert MODULE._rate_label(-0.6) == "Decreasing"
    assert MODULE._rate_label(0.2) == "Stable"


def test_recommended_window_uses_earliest_ready_horizon() -> None:
    row = pd.Series(
        {
            "classifier_derived_hui": 35.0,
            "predicted_hui_24h": 55.0,
            "predicted_hui_48h": 65.0,
            "predicted_hui_72h": 70.0,
        }
    )
    window, _ = MODULE._recommended_window(row)
    assert window == "Within 24–48 hours"


def test_limited_calibration_caps_confidence_at_moderate() -> None:
    score, label = MODULE._confidence(
        calibration_gate={
            "gate_passed": False,
            "selected_method": "platt",
        },
        hrsi=100.0,
        completeness=100.0,
    )
    assert score == 74.9
    assert label == "Moderate"


def test_passed_calibration_can_reach_high_confidence() -> None:
    score, label = MODULE._confidence(
        calibration_gate={
            "gate_passed": True,
            "selected_method": "platt",
        },
        hrsi=100.0,
        completeness=100.0,
    )
    assert score == 100.0
    assert label == "High"
