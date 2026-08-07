from __future__ import annotations

import numpy as np
import pandas as pd

from multivari.common.schema import HIVE_COLUMN, TIMESTAMP_COLUMN
from multivari.common.splitting import assign_chronological_splits
from multivari.modules.absconding.config import AbscondingSettings
from multivari.modules.absconding.events import build_event_episodes
from multivari.modules.absconding.features import build_absconding_features, select_feature_columns
from multivari.modules.absconding.metrics import choose_alert_threshold
from multivari.modules.absconding.pipeline import prepare_absconding_dataset


def synthetic_frame(hours: int = 800) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=hours, freq="h")
    rows = []
    for hive_index, hive_id in enumerate(("hive-a", "hive-b")):
        event = np.zeros(hours, dtype=np.int8)
        first_event = min(hours - 20, max(30, hours // 2 + hive_index * 10))
        event[first_event] = 1
        event[min(hours - 1, first_event + 6)] = 1
        rows.append(
            pd.DataFrame(
                {
                    TIMESTAMP_COLUMN: timestamps,
                    HIVE_COLUMN: hive_id,
                    "temperature_c": 33 + np.sin(np.arange(hours) / 24),
                    "co2_ppm": 700 + np.arange(hours) * 0.1 + hive_index,
                    "humidity_pct": 60 + np.cos(np.arange(hours) / 48),
                    "weight_kg": 35 - np.arange(hours) * 0.002,
                    "external_temperature_c": 27 + np.sin(np.arange(hours) / 24),
                    "external_humidity_pct": 70 + np.cos(np.arange(hours) / 24),
                    "brood_health_healthy_1": 1,
                    "swarming_happened_1": 0,
                    "absconding_happened_1": event,
                    "honey_harvested_1": 0,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_event_markers_are_merged_into_episodes() -> None:
    frame = synthetic_frame()
    episodes = build_event_episodes(
        frame,
        event_column="absconding_happened_1",
        merge_gap_hours=24,
    )
    assert len(episodes) == 2
    assert set(episodes["marker_count"]) == {2}


def test_feature_selection_excludes_all_label_columns() -> None:
    settings = AbscondingSettings(minimum_history_hours=24)
    featured = build_absconding_features(synthetic_frame(200), settings)
    featured[settings.target_column] = 0
    columns = select_feature_columns(
        featured,
        extra_excluded={settings.target_column, settings.event_column},
    )
    assert settings.target_column not in columns
    assert settings.event_column not in columns
    assert "brood_health_healthy_1" not in columns
    assert "swarming_happened_1" not in columns
    assert "honey_harvested_1" not in columns
    assert "weight_kg_change_24h" in columns
    assert "multisensor_instability_index" in columns
    assert "internal_external_temperature_difference" in columns
    assert "internal_external_humidity_difference" in columns


def test_prepared_dataset_uses_future_target_and_boundary_gaps() -> None:
    frame = synthetic_frame(1000)
    manifest = assign_chronological_splits(
        frame,
        train_fraction=0.6,
        validation_fraction=0.2,
        boundary_gap_hours=72,
    )
    settings = AbscondingSettings(minimum_history_hours=24)
    prepared, feature_names, _episodes = prepare_absconding_dataset(frame, manifest, settings)
    assert not prepared["is_boundary_gap"].any()
    assert prepared[settings.target_column].isin([0, 1]).all()
    assert settings.target_column not in feature_names
    assert settings.event_column not in feature_names
    event_time = frame.loc[
        (frame[HIVE_COLUMN] == "hive-a") & frame["absconding_happened_1"].eq(1),
        TIMESTAMP_COLUMN,
    ].min()
    warning_rows = prepared.loc[
        prepared[HIVE_COLUMN].eq("hive-a")
        & prepared[TIMESTAMP_COLUMN].ge(
            event_time - pd.Timedelta(hours=settings.prediction_horizon_hours)
        )
        & prepared[TIMESTAMP_COLUMN].lt(event_time)
    ]
    assert warning_rows[settings.target_column].eq(1).all()


def test_threshold_selection_respects_alert_fraction_when_possible() -> None:
    y_true = np.array([0] * 96 + [1] * 4)
    probability = np.linspace(0, 1, 100)
    result = choose_alert_threshold(
        y_true,
        probability,
        beta=2.0,
        maximum_alert_fraction=0.10,
    )
    assert result["constraint_satisfied"] is True
    assert result["alert_fraction"] <= 0.10
    assert 0 <= result["threshold"] <= 1
