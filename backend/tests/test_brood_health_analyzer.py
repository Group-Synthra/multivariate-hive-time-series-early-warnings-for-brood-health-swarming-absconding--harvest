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
    frame = pd.DataFrame(
        {
            "hive_id": ["h1"] * 12,
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="h"),
            "temperature_c": [35.0] * 12,
            "humidity_pct": [65.0] * 12,
            "co2_ppm": [2500.0] * 12,
            "weight_kg": [30.0] * 12,
        }
    )
    result = compute_condition_history(frame)
    assert result["condition_score"].between(0, 100).all()
    assert result["bhsi"].between(0, 100).all()


def test_exact_and_safety_scores_produce_critical_warning():
    warning = build_warning_payload(
        exact_forecast_score=15,
        safety_minimum_score=12,
        current_condition_score=30,
        bhsi=20,
        rod_points_per_hour=-3.1,
        exact_forecast_drop_points=15,
        safety_drop_points=18,
        domain_shift_warnings=[],
        history_sufficient=True,
    )
    assert warning["level"] == "Critical Alert"
    assert warning["requires_physical_confirmation"] is True


def test_good_future_health_can_have_separate_warning_severity():
    warning = build_warning_payload(
        exact_forecast_score=65.30,
        safety_minimum_score=65.30,
        current_condition_score=77.50,
        forecast_bhsi=38.80,
        forecast_rod_points_per_hour=-1.41,
        exact_forecast_drop_points=12.20,
        safety_drop_points=12.20,
        domain_shift_warnings=[],
        history_sufficient=True,
    )
    assert warning["predicted_health_level"] == "Good"
    assert warning["level"] == "Warning"
    assert warning["title"] == "Deterioration warning"
    assert "65.30/100 (Good)" in warning["summary"]


def test_low_store_signal_adds_conditional_feeding_action():
    warning = build_warning_payload(
        exact_forecast_score=55.0,
        safety_minimum_score=52.0,
        current_condition_score=63.0,
        forecast_bhsi=52.0,
        forecast_rod_points_per_hour=-1.0,
        exact_forecast_drop_points=8.0,
        safety_drop_points=11.0,
        weight_change_pct_24h=-4.5,
        weight_component=35.0,
        domain_shift_warnings=[],
        history_sufficient=True,
    )
    action_text = " ".join(warning["recommended_actions"]).lower()
    assert "sucrose syrup" in action_text
    assert "stores are genuinely low" in action_text


def test_domain_shift_is_confidence_note_not_alert_escalation():
    warning = build_warning_payload(
        exact_forecast_score=88.0,
        safety_minimum_score=85.0,
        current_condition_score=90.0,
        forecast_bhsi=85.0,
        forecast_rod_points_per_hour=0.0,
        exact_forecast_drop_points=2.0,
        safety_drop_points=5.0 - 3.0,
        domain_shift_warnings=["weight scale differs"],
        history_sufficient=True,
    )
    assert warning["level"] == "Normal"
    assert warning["confidence_notes"] == ["weight scale differs"]
