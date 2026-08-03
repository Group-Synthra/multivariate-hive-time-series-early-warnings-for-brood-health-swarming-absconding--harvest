import pandas as pd

from multivari.common.cleaning import clean_common_dataset


def test_cleaning_sorts_and_deduplicates() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 01:00", "2024-01-01 00:00", "2024-01-01 00:00"],
            "hive_id": [" hive1 ", "hive1", "hive1"],
            "temperature_c": [35, 34, 34],
            "co2_ppm": [800, 700, 700],
            "humidity_pct": [60, 61, 61],
            "weight_kg": [40, 40, 40],
            "brood_health_healthy_1": [1, 1, 1],
            "swarming_happened_1": [0, 0, 0],
            "absconding_happened_1": [0, 0, 0],
            "honey_harvested_1": [0, 0, 0],
        }
    )
    clean = clean_common_dataset(frame)
    assert len(clean) == 2
    assert clean.iloc[0]["hive_id"] == "hive1"
