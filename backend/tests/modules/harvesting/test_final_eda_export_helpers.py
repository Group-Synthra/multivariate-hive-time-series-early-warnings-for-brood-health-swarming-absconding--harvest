from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/export_final_harvest_eda_dashboard.py"
SPEC = importlib.util.spec_from_file_location("final_eda_export", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sensor_family_excludes_calendar_features() -> None:
    assert MODULE._feature_family("weight_delta_72h_kg") == "weight"
    assert MODULE._feature_family("humidity_pct_delta_24h") == "humidity"
    assert MODULE._feature_family("temperature_c_delta_24h") == "temperature"
    assert MODULE._feature_family("co2_ppm_delta_24h") == "co2"
    assert MODULE._feature_family("day_of_week_cos") is None


def test_signal_evolution_uses_largest_family_effect() -> None:
    frame = pd.DataFrame(
        {
            "lead_hours": [72, 72, 72, 24, 24],
            "feature": [
                "weight_delta_72h_kg",
                "weight_std_168h_kg",
                "humidity_pct_delta_24h",
                "weight_delta_72h_kg",
                "temperature_c_delta_24h",
            ],
            "absolute_standardized_mean_difference": [1.2, 0.9, 1.3, 0.8, 1.1],
        }
    )
    result = MODULE._signal_evolution(frame, [72, 24])
    assert result[0]["weight"] == 1.2
    assert result[0]["humidity"] == 1.3
    assert result[1]["temperature"] == 1.1
