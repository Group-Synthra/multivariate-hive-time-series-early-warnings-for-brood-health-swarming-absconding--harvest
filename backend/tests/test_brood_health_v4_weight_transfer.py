import pandas as pd

from multivari.modules.brood_health.features import map_iot_frame


def test_live_weight_conversion_is_applied_before_relative_features() -> None:
    raw = pd.DataFrame(
        {
            "hive_id": ["hive-a", "hive-a"],
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
            "temperature_c": [34.0, 34.2],
            "humidity_pct": [65.0, 66.0],
            "co2_ppm": [800.0, 850.0],
            "weight_kg": [4.0, 4.1],
        }
    )
    mapped = map_iot_frame(
        raw,
        weight_scale_factor=10.0,
        weight_offset_kg=2.0,
    )
    assert mapped["weight_kg"].tolist() == [42.0, 43.0]
