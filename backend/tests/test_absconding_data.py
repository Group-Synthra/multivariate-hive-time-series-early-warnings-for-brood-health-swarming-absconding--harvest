from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from multivari.common.io import read_table
from multivari.common.schema import HIVE_COLUMN, TIMESTAMP_COLUMN
from multivari.modules.absconding.config import AbscondingSettings
from multivari.modules.absconding.data import (
    clean_absconding_source,
    run_absconding_data_pipeline,
)


def source_frame(hours: int = 400) -> pd.DataFrame:
    rows = []
    for hive_id, event_start in (("hive-a", 220), ("hive-b", 280)):
        active = [0] * hours
        active[event_start : event_start + 4] = [1, 1, 1, 1]
        for index, timestamp in enumerate(pd.date_range("2025-01-01", periods=hours, freq="h")):
            rows.append(
                {
                    "timestamp": timestamp,
                    "hive_id": hive_id,
                    "internal_temperature_c": 34.0 + index * 0.001,
                    "internal_humidity_pct": 62.0,
                    "co2_ppm": 900.0 + index,
                    "hive_weight_kg": 35.0 - index * 0.002,
                    "external_temperature_c": 28.0,
                    "external_humidity_pct": 70.0,
                    "absconding_event_label": active[index],
                    "absconding_label_next_72h": 0,
                    "swarming_event_label": 1,
                }
            )
    return pd.DataFrame(rows)


def test_source_mapping_uses_event_onsets_and_drops_other_labels() -> None:
    settings = AbscondingSettings()
    clean, _diagnostics = clean_absconding_source(source_frame(), settings)
    assert clean[settings.active_event_column].sum() == 8
    assert clean[settings.event_column].sum() == 2
    assert "swarming_event_label" not in clean
    assert "absconding_label_next_72h" not in clean
    assert "external_temperature_c" in clean
    assert "external_humidity_pct" in clean


def test_module_data_pipeline_writes_separate_outputs(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "absconding.csv"
    source_frame(500).to_csv(source, index=False)
    config = {
        "data": {
            "input_path": str(source),
            "clean_path": "data/processed/absconding_clean.parquet",
            "manifest_path": "data/manifests/absconding_split_manifest.parquet",
            "profile_path": "artifacts/reports/absconding/data_profile.json",
            "columns": {
                "timestamp": "timestamp",
                "hive": "hive_id",
                "temperature": "internal_temperature_c",
                "humidity": "internal_humidity_pct",
                "co2": "co2_ppm",
                "weight": "hive_weight_kg",
                "external_temperature": "external_temperature_c",
                "external_humidity": "external_humidity_pct",
                "event": "absconding_event_label",
                "precomputed_72h_target": "absconding_label_next_72h",
            },
        },
        "split": {
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "boundary_gap_hours": 24,
        },
        "module": {
            "minimum_history_hours": 24,
            "prediction_horizon_hours": 24,
        },
    }
    config_path = tmp_path / "absconding.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    profile = run_absconding_data_pipeline(
        input_path=source,
        backend_root=tmp_path,
        config_path=config_path,
    )
    clean_path = tmp_path / "data/processed/absconding_clean.parquet"
    manifest_path = tmp_path / "data/manifests/absconding_split_manifest.parquet"
    assert clean_path.is_file()
    assert manifest_path.is_file()
    clean = read_table(clean_path)
    manifest = read_table(manifest_path)
    assert profile.event_onset_markers == 2
    assert profile.merged_event_episodes == 2
    assert set(manifest["split"]) == {"train", "validation", "test"}
    assert clean[HIVE_COLUMN].nunique() == 2
    assert pd.api.types.is_datetime64_any_dtype(clean[TIMESTAMP_COLUMN])
