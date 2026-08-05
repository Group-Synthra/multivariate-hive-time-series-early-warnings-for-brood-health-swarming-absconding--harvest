import numpy as np
import pandas as pd

from multivari.modules.brood_health.features import (
    TARGET_COLUMN,
    build_feature_frame,
    build_supervised_dataset,
)
from multivari.modules.brood_health.scoring import (
    classify_health_level,
    compute_score_components,
)
from multivari.modules.brood_health.training import _assign_hive_splits


def make_frame(hives: int = 6, hours: int = 120) -> pd.DataFrame:
    rows = []
    for hive_index in range(hives):
        timestamps = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
        for position, timestamp in enumerate(timestamps):
            deterioration = position >= 96 and hive_index % 2 == 0
            rows.append(
                {
                    "hive_id": f"hive-{hive_index}",
                    "timestamp": timestamp,
                    "temperature_c": 27.0 if deterioration else 35.0,
                    "humidity_pct": 82.0 if deterioration else 65.0,
                    "co2_ppm": 12_000.0 if deterioration else 2_500.0,
                    "weight_kg": 30.0 - (0.03 * max(0, position - 96) if deterioration else 0.0),
                    TARGET_COLUMN: 0 if deterioration else 1,
                }
            )
    return pd.DataFrame(rows)


def test_score_range_and_level_boundaries():
    scored = compute_score_components(make_frame(hives=1))
    assert scored["brood_health_score"].between(1, 100).all()
    assert classify_health_level(39.9) == "Critical"
    assert classify_health_level(40) == "Poor"
    assert classify_health_level(60) == "Good"
    assert classify_health_level(80) == "Excellent"


def test_future_target_is_a_continuous_future_window_minimum():
    frame = make_frame(hives=1)
    x, y, metadata, columns = build_supervised_dataset(frame, horizon_hours=6)
    assert len(x) == len(y) == len(metadata)
    assert y.between(1, 100).all()
    assert (metadata["target_timestamp"] > metadata["timestamp"]).all()
    assert TARGET_COLUMN not in columns
    assert "hive_id" not in columns
    assert "day_of_year_sin" not in columns
    assert metadata["transition_window"].any()


def test_future_rows_never_change_earlier_features():
    base = make_frame(hives=1)
    changed = base.copy()
    changed.loc[changed.index >= 105, "temperature_c"] = -5.0
    before = build_feature_frame(base).iloc[100]
    after = build_feature_frame(changed).iloc[100]
    pd.testing.assert_series_equal(before, after)


def test_group_split_never_places_one_hive_in_multiple_partitions():
    frame = make_frame(hives=10)
    _, y, metadata, _ = build_supervised_dataset(frame, horizon_hours=6)
    split = _assign_hive_splits(metadata, y)
    audit = pd.DataFrame({"hive_id": metadata["hive_id"], "split": split})
    assert audit.groupby("hive_id")["split"].nunique().max() == 1
    assert set(audit["split"].unique()) == {"train", "validation", "test"}


def test_binary_persistence_audit_exposes_stable_label_baseline():
    from multivari.modules.brood_health.audit import binary_target_persistence_audit

    audit = binary_target_persistence_audit(make_frame(hives=2), horizons=(1, 6))
    by_horizon = {row["horizon_hours"]: row for row in audit["horizons"]}
    assert by_horizon[1]["persistence_accuracy"] > 0.95
    assert by_horizon[6]["transition_rows"] > 0


def test_feature_schema_leakage_audit_rejects_target_and_identifiers():
    from multivari.modules.brood_health.audit import feature_leakage_audit

    safe = feature_leakage_audit(["temperature_c", "co2_ppm_lag_6h", "weight_change_pct_24h"])
    assert safe["passed"] is True

    unsafe = feature_leakage_audit(["temperature_c", TARGET_COLUMN, "hive_id", "future_temperature"])
    assert unsafe["passed"] is False
    assert TARGET_COLUMN in unsafe["target_columns_in_features"]
    assert "hive_id" in unsafe["hive_identifier_features"]
    assert "future_temperature" in unsafe["future_sensor_values_in_features"]
