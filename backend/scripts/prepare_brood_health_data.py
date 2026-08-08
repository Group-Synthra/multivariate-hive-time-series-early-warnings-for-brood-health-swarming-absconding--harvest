from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from multivari.modules.brood_health.config import PATHS
from multivari.modules.brood_health.features import (
    EXTERNAL_SENSORS,
    HISTORICAL_FEATURE_TIMEZONE,
    build_feature_frame,
    normalise_historical,
)


def prepare_brood_health_dataset(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    source = Path(input_path or PATHS.module_workbook)
    destination = Path(output_path or PATHS.module_processed)

    if not source.exists():
        raise FileNotFoundError(f"Brood Health workbook not found: {source}")

    raw = pd.read_excel(source, sheet_name="Common_Dataset")
    normalized = normalise_historical(
        raw,
        naive_timezone=HISTORICAL_FEATURE_TIMEZONE,
    )

    missing_external = {
        column: int(normalized[column].isna().sum())
        for column in EXTERNAL_SENSORS
    }
    if any(value == len(normalized) for value in missing_external.values()):
        raise ValueError(
            "The Brood Health workbook must contain external temperature and "
            "external humidity values before module preprocessing."
        )

    local_timestamp = normalized["timestamp"].dt.tz_convert(
        HISTORICAL_FEATURE_TIMEZONE
    )
    normalized["source_local_timestamp"] = local_timestamp.dt.tz_localize(None)
    normalized["source_timezone"] = HISTORICAL_FEATURE_TIMEZONE

    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(destination, index=False)

    feature_frame = build_feature_frame(
        normalized,
        feature_timezone=HISTORICAL_FEATURE_TIMEZONE,
    )

    audit = {
        "source": str(source),
        "output": str(destination),
        "rows": len(normalized),
        "hives": int(normalized["hive_id"].nunique()),
        "timestamp_storage": "UTC",
        "historical_local_timezone": HISTORICAL_FEATURE_TIMEZONE,
        "first_timestamp_utc": normalized["timestamp"].min().isoformat(),
        "last_timestamp_utc": normalized["timestamp"].max().isoformat(),
        "external_missing": missing_external,
        "feature_count": int(feature_frame.shape[1]),
        "contains_external_temperature_features": any(
            column.startswith("external_temp") for column in feature_frame.columns
        ),
        "contains_external_humidity_features": any(
            column.startswith("external_humidity") for column in feature_frame.columns
        ),
    }
    PATHS.module_preprocessing_audit.parent.mkdir(parents=True, exist_ok=True)
    PATHS.module_preprocessing_audit.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    audit = prepare_brood_health_dataset(args.input, args.output)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
