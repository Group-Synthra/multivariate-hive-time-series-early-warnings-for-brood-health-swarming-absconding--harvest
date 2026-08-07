import numpy as np
import pandas as pd

from multivari.modules.harvesting.label_audit import (
    audit_harvest_event_alignment,
)


def test_detects_delayed_marker_after_sustained_drop() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=120,
        freq="h",
    )
    weights = np.full(120, 50.0)
    weights[80:] = 44.0

    common = pd.DataFrame(
        {
            "hive_id": ["h1"] * 120,
            "timestamp": timestamps,
            "weight_kg": weights,
            "co2_ppm": np.linspace(
                1000,
                1200,
                120,
            ),
        }
    )
    events = pd.DataFrame(
        {
            "hive_id": ["h1"],
            "harvest_event_id": ["h1_harvest_001"],
            "event_start": [timestamps[85]],
            "split": ["train"],
        }
    )

    audit = audit_harvest_event_alignment(
        common,
        events,
        lookback_hours=24,
        minimum_drop_kg=2.0,
        mad_multiplier=6.0,
        minimum_persistent_drop_kg=1.5,
        aligned_tolerance_hours=2,
    )

    assert audit.loc[0, "alignment_status"] == "marker_delayed"
    assert audit.loc[0, "marker_delay_hours"] == 5.0


def test_flags_flatline_co2_before_event() -> None:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=120,
        freq="h",
    )
    weights = np.full(120, 50.0)
    weights[90:] = 46.0

    common = pd.DataFrame(
        {
            "hive_id": ["h1"] * 120,
            "timestamp": timestamps,
            "weight_kg": weights,
            "co2_ppm": np.full(120, 1200.0),
        }
    )
    events = pd.DataFrame(
        {
            "hive_id": ["h1"],
            "harvest_event_id": ["h1_harvest_001"],
            "event_start": [timestamps[90]],
        }
    )

    audit = audit_harvest_event_alignment(
        common,
        events,
    )

    assert audit.loc[0, "co2_flatline_pre72h"] == 1
