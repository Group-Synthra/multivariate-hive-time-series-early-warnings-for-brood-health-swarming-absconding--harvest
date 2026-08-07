from multivari.modules.brood_health.analyzer import (
    classify_stability,
    classify_trend,
)
from multivari.modules.brood_health.scoring import classify_health_level


def test_non_overlapping_health_boundaries() -> None:
    assert classify_health_level(1) == "Critical"
    assert classify_health_level(39.999) == "Critical"
    assert classify_health_level(40) == "Poor"
    assert classify_health_level(59.999) == "Poor"
    assert classify_health_level(60) == "Good"
    assert classify_health_level(79.999) == "Good"
    assert classify_health_level(80) == "Excellent"
    assert classify_health_level(100) == "Excellent"


def test_bhsi_and_rod_boundaries_are_consistent() -> None:
    assert classify_stability(39.999) == "Low"
    assert classify_stability(40) == "Moderate"
    assert classify_stability(69.999) == "Moderate"
    assert classify_stability(70) == "High"

    assert classify_trend(-3.01) == "Rapid Declining"
    assert classify_trend(-3.0) == "Slow Declining"
    assert classify_trend(-0.5) == "Stable"
    assert classify_trend(0.5) == "Stable"
    assert classify_trend(0.51) == "Slow Improving"
    assert classify_trend(3.01) == "Rapid Improving"
