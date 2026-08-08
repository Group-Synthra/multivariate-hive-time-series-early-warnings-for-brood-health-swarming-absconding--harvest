from pathlib import Path
from runpy import run_path

import pandas as pd

_EXPORTER = run_path(
    str(
        Path(__file__).resolve().parents[3] / "scripts" / "export_harvesting_benchmark_dashboard.py"
    )
)
_prepare_24h_series = _EXPORTER["_prepare_24h_series"]


def test_prepare_24h_series_keeps_latest_rows_per_hive() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": list(
                pd.date_range(
                    "2024-01-01",
                    periods=5,
                    freq="h",
                )
            )
            * 2,
            "hive_id": ["h1"] * 5 + ["h2"] * 5,
            "current_weight_kg": list(range(5)) + list(range(10, 15)),
            "predicted_future_weight_kg": list(range(1, 6)) + list(range(11, 16)),
            "actual_future_weight_kg": list(range(1, 6)) + list(range(11, 16)),
            "predicted_delta_kg": [1.0] * 10,
            "actual_delta_kg": [1.0] * 10,
        }
    )

    records, hives = _prepare_24h_series(
        frame,
        rows_per_hive=2,
    )

    assert hives == ["h1", "h2"]
    assert len(records) == 4
    assert {record["timestamp"] for record in records if record["hive_id"] == "h1"} == {
        pd.Timestamp("2024-01-01 03:00").isoformat(),
        pd.Timestamp("2024-01-01 04:00").isoformat(),
    }
