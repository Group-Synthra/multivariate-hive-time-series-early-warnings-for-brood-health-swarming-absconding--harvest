from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from multivari.common.schema import HIVE_COLUMN, TIMESTAMP_COLUMN
from multivari.modules.absconding.lstm import (
    build_sequence_set,
    preferred_lstm_features,
)


def test_preferred_lstm_features_keep_report_aligned_inputs() -> None:
    available = [
        "temperature_c",
        "humidity_pct",
        "co2_ppm",
        "weight_kg",
        "temperature_c_change_24h",
        "weight_kg_roll_mean_72h",
        "environmental_stress_score",
        "unrelated_numeric_feature",
    ]
    selected = preferred_lstm_features(available)
    assert "temperature_c" in selected
    assert "temperature_c_change_24h" in selected
    assert "weight_kg_roll_mean_72h" in selected
    assert "environmental_stress_score" in selected
    assert "unrelated_numeric_feature" not in selected


def test_sequence_builder_preserves_positive_endpoints_when_capped() -> None:
    rows = 40
    frame = pd.DataFrame(
        {
            HIVE_COLUMN: ["hive-a"] * rows,
            TIMESTAMP_COLUMN: pd.date_range("2026-01-01", periods=rows, freq="h"),
            "split": ["train"] * rows,
            "target": [0] * 30 + [1] + [0] * 9,
            "temperature_c": np.linspace(32.0, 36.0, rows),
            "humidity_pct": np.linspace(55.0, 65.0, rows),
        }
    )
    features = ["temperature_c", "humidity_pct"]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    scaler.fit(imputer.fit_transform(frame[features]))

    sequence_set = build_sequence_set(
        frame,
        feature_names=features,
        target_column="target",
        sequence_length=8,
        stride=4,
        imputer=imputer,
        scaler=scaler,
        maximum_sequences=4,
        random_state=42,
    )

    assert sequence_set.X.shape == (4, 8, 2)
    assert sequence_set.y.sum() == 1
    positive_time = sequence_set.metadata.loc[
        sequence_set.y.astype(bool), TIMESTAMP_COLUMN
    ].iloc[0]
    assert positive_time == frame.loc[30, TIMESTAMP_COLUMN]
