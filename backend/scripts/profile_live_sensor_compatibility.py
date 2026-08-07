from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = BACKEND_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from multivari.iot.postgres_repository import (
    PostgresSensorRepository,
    PostgresSensorSettings,
)

SENSORS = {
    "weight_kg": {
        "live_column": "total_weight",
        "training_column": "weight_kg_current",
        "scale_setting": "weight_scale",
        "unit": "kg",
    },
    "internal_temperature_c": {
        "live_column": "internal_temperature",
        "training_column": "temperature_c_current",
        "scale_setting": "temperature_scale",
        "unit": "°C",
    },
    "internal_humidity_pct": {
        "live_column": "internal_humidity",
        "training_column": "humidity_pct_current",
        "scale_setting": "humidity_scale",
        "unit": "%",
    },
    "co2_ppm": {
        "live_column": "internal_co2",
        "training_column": "co2_ppm_current",
        "scale_setting": "co2_scale",
        "unit": "ppm",
    },
}


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def main() -> None:
    load_dotenv(BACKEND_ROOT / ".env")
    settings = PostgresSensorSettings.from_env()
    repository = PostgresSensorRepository(settings)
    live = repository.fetch_recent()

    training_path = (
        BACKEND_ROOT
        / "data/processed/harvest_reviewed_feature_dataset.parquet"
    )
    if not training_path.exists():
        raise FileNotFoundError(
            f"Training feature dataset is missing: {training_path}"
        )
    training = pd.read_parquet(training_path)
    if "split" in training.columns:
        training = training.loc[training["split"].eq("train")]

    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for name, config in SENSORS.items():
        live_column = str(config["live_column"])
        training_column = str(config["training_column"])
        scale = float(getattr(settings, str(config["scale_setting"])))

        if live_column not in live.columns:
            records.append(
                {
                    "sensor": name,
                    "status": "live_column_missing",
                    "live_column": live_column,
                    "training_column": training_column,
                }
            )
            warnings.append(f"Live column is missing for {name}: {live_column}")
            continue
        if training_column not in training.columns:
            records.append(
                {
                    "sensor": name,
                    "status": "training_column_missing",
                    "live_column": live_column,
                    "training_column": training_column,
                }
            )
            continue

        live_values = pd.to_numeric(live[live_column], errors="coerce") * scale
        train_values = pd.to_numeric(training[training_column], errors="coerce")
        live_values = live_values.dropna()
        train_values = train_values.dropna()

        if live_values.empty or train_values.empty:
            records.append(
                {
                    "sensor": name,
                    "status": "no_numeric_values",
                    "live_column": live_column,
                    "training_column": training_column,
                }
            )
            continue

        train_q01 = float(train_values.quantile(0.01))
        train_q99 = float(train_values.quantile(0.99))
        live_median = float(live_values.median())
        outside_fraction = float(
            ((live_values < train_q01) | (live_values > train_q99)).mean()
        )
        median_inside = train_q01 <= live_median <= train_q99
        status = "compatible_range" if median_inside else "domain_shift_warning"
        if not median_inside:
            warnings.append(
                f"{name} live median {live_median:.3f} is outside the training "
                f"1st–99th percentile range [{train_q01:.3f}, {train_q99:.3f}]."
            )

        records.append(
            {
                "sensor": name,
                "status": status,
                "unit": config["unit"],
                "scale": scale,
                "live_rows": len(live_values),
                "live_minimum": _safe_float(live_values.min()),
                "live_median": live_median,
                "live_maximum": _safe_float(live_values.max()),
                "training_rows": len(train_values),
                "training_q01": train_q01,
                "training_median": float(train_values.median()),
                "training_q99": train_q99,
                "live_fraction_outside_training_q01_q99": outside_fraction,
            }
        )

    payload = {
        "status": "compatible" if not warnings else "review_required",
        "important_notice": (
            "This check compares units and numeric ranges only. It cannot prove "
            "that the Sri Lankan IoT box measures the same biological quantity "
            "as the historical research hives."
        ),
        "records": records,
        "warnings": warnings,
    }

    output_path = (
        BACKEND_ROOT
        / "artifacts/reports/harvesting/live_iot_sensor_compatibility.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "output_path": str(output_path)}, indent=2))



if __name__ == "__main__":
    main()
