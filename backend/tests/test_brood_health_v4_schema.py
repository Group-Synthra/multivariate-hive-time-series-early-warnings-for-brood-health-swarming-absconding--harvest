import numpy as np
import pandas as pd

from multivari.modules.brood_health.features import build_supervised_dataset


def test_forecast_features_exclude_labels_identifiers_future_values_and_absolute_weight() -> None:
    hours = 110
    frame = pd.DataFrame(
        {
            "hive_id": ["hive-a"] * hours,
            "timestamp": pd.date_range("2024-01-01", periods=hours, freq="h"),
            "temperature_c": np.linspace(33, 36, hours),
            "humidity_pct": np.linspace(60, 70, hours),
            "co2_ppm": np.linspace(700, 1500, hours),
            "weight_kg": np.linspace(35, 40, hours),
            "brood_health_healthy_1": [1] * hours,
        }
    )
    _, _, _, columns = build_supervised_dataset(frame, horizon_hours=6)
    lowered = [column.lower() for column in columns]
    assert "weight_kg" not in columns
    assert all("brood_health" not in column for column in lowered)
    assert all("healthy_1" not in column for column in lowered)
    assert all("future" not in column for column in lowered)
    assert all("target" not in column for column in lowered)
    assert all("hive_id" not in column for column in lowered)
    assert all("timestamp" not in column for column in lowered)
    assert any(column.startswith("weight_change_pct_") for column in columns)
    assert any(column.startswith("weight_cv_") for column in columns)
