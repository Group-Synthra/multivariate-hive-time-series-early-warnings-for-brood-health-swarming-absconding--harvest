from __future__ import annotations

import json
from pathlib import Path
import yaml

from .cleaning import clean_common_dataset
from .eda import generate_common_eda
from .io import read_table, write_parquet
from .splitting import assign_chronological_splits
from .validation import profile_and_validate


def run_common_pipeline(
    *,
    input_path: str | Path,
    config_path: str | Path,
    processed_directory: str | Path,
    report_directory: str | Path,
    manifest_directory: str | Path,
) -> None:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    processed = Path(processed_directory)
    reports = Path(report_directory)
    manifests = Path(manifest_directory)
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    raw = read_table(input_path, sheet_name=config["dataset"]["sheet_name"])
    initial_report = profile_and_validate(raw)
    (reports / "raw_validation_report.json").write_text(
        json.dumps(initial_report.to_dict(), indent=2), encoding="utf-8"
    )

    cleaning = config["cleaning"]
    clean = clean_common_dataset(
        raw,
        interpolate_missing_sensors=cleaning["interpolate_missing_sensors"],
        interpolation_limit_rows=cleaning["interpolation_limit_rows"],
    )
    clean_report = profile_and_validate(clean)
    (reports / "clean_validation_report.json").write_text(
        json.dumps(clean_report.to_dict(), indent=2), encoding="utf-8"
    )

    write_parquet(clean, processed / "common_clean.parquet")
    generate_common_eda(clean, reports / "common_eda")

    split_config = config["split"]
    split_manifest = assign_chronological_splits(
        clean,
        train_fraction=split_config["train_fraction"],
        validation_fraction=split_config["validation_fraction"],
        boundary_gap_hours=split_config["boundary_gap_hours"],
    )
    write_parquet(split_manifest, manifests / "common_split_manifest.parquet")
