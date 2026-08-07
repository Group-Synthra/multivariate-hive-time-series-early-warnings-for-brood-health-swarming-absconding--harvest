import numpy as np
import pandas as pd

from multivari.modules.harvesting.eda import (
    add_research_eda_features,
    build_event_lead_samples,
    calculate_variable_relationships,
)


def _base_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=220, freq="h")
    return pd.DataFrame(
        {
            "hive_id": ["h1"] * 220,
            "timestamp": timestamps,
            "weight_kg": np.arange(220, dtype=float),
            "temperature_c": [34.0] * 220,
            "humidity_pct": [65.0] * 220,
            "co2_ppm": [1200.0] * 220,
        }
    )


def test_features_do_not_cross_hive_boundaries() -> None:
    first = _base_frame().iloc[:40].copy()
    second = first.copy()
    second["hive_id"] = "h2"
    second["weight_kg"] = second["weight_kg"] + 1000
    result = add_research_eda_features(pd.concat([first, second], ignore_index=True))
    first_h2 = result.loc[result["hive_id"].eq("h2")].iloc[0]
    assert pd.isna(first_h2["weight_change_24h"])


def test_event_lead_samples_use_requested_leads() -> None:
    frame = add_research_eda_features(_base_frame())
    event_start = frame.loc[180, "timestamp"]
    events = pd.DataFrame(
        {
            "hive_id": ["h1"],
            "harvest_event_id": ["h1_harvest_001"],
            "event_start": [event_start],
            "split": ["train"],
        }
    )
    samples = build_event_lead_samples(frame, events, lead_hours=[72, 24, 1])
    assert set(samples["lead_hours"]) == {72, 24, 1}
    assert len(samples) == 3


def test_relationship_effect_direction() -> None:
    events = pd.DataFrame({"lead_hours": [72, 72, 72]})
    controls = pd.DataFrame({"lead_hours": [72, 72, 72]})
    for feature in [
        "weight_change_24h",
        "weight_change_72h",
        "weight_change_168h",
        "distance_from_max_168h",
        "relative_to_max_168h",
        "weight_std_24h",
        "temperature_mean_24h",
        "temperature_std_24h",
        "temperature_change_24h",
        "humidity_mean_24h",
        "humidity_std_24h",
        "humidity_change_24h",
        "co2_mean_24h",
        "co2_std_24h",
        "co2_change_24h",
        "environmental_stability_24h",
    ]:
        events[feature] = np.nan
        controls[feature] = np.nan
    events["weight_change_72h"] = [3.0, 4.0, 5.0]
    controls["weight_change_72h"] = [-1.0, 0.0, 1.0]
    relationships = calculate_variable_relationships(events, controls)
    row = relationships.loc[
        relationships["feature"].eq("weight_change_72h")
    ].iloc[0]
    assert row["cliffs_delta"] > 0
