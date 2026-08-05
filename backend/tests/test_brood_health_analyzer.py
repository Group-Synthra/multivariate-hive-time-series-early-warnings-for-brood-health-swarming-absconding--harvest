import pandas as pd

from multivari.modules.brood_health.analyzer import (
    build_warning_payload,
    classify_health_level,
    compute_condition_history,
)


def test_health_level_boundaries():
    assert classify_health_level(0) == "Critical"
    assert classify_health_level(39.99) == "Critical"
    assert classify_health_level(40) == "Poor"
    assert classify_health_level(60) == "Good"
    assert classify_health_level(80) == "Excellent"
    assert classify_health_level(100) == "Excellent"


def test_condition_history_stays_in_range():
    frame = pd.DataFrame({
        "hive_id": ["h1"] * 12,
        "timestamp": pd.date_range("2026-01-01", periods=12, freq="h"),
        "temperature_c": [35.0] * 12,
        "humidity_pct": [65.0] * 12,
        "co2_ppm": [2500.0] * 12,
        "weight_kg": [30.0] * 12,
    })
    result = compute_condition_history(frame)
    assert result["condition_score"].between(0, 100).all()
    assert result["bhsi"].between(0, 100).all()


def test_high_unhealthy_probability_produces_critical_warning():
    warning = build_warning_payload(
        forecast_score=15,
        current_condition_score=30,
        bhsi=20,
        rod_points_per_hour=-3,
        unhealthy_probability=0.9,
    )
    assert warning["level"] == "Critical"
    assert warning["requires_physical_confirmation"] is True
