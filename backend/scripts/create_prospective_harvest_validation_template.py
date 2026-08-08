from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TEMPLATE_COLUMNS = [
    "record_id",
    "hive_id",
    "apiary_id",
    "inspection_timestamp",
    "beekeeper_id",
    "harvest_decision",
    "planned_harvest_timestamp",
    "actual_harvest_timestamp",
    "pre_harvest_hive_weight_kg",
    "post_harvest_hive_weight_kg",
    "harvested_honey_mass_kg",
    "comb_capping_percent",
    "honey_moisture_percent",
    "nectar_flow_observed",
    "weather_notes",
    "colony_health_notes",
    "sensor_quality_notes",
    "harvest_confirmed",
    "decision_notes",
]


DATA_DICTIONARY = [
    {
        "column": "record_id",
        "description": "Unique prospective validation record.",
        "required": True,
        "allowed_values": "unique text",
    },
    {
        "column": "hive_id",
        "description": "Hive identifier matching telemetry data.",
        "required": True,
        "allowed_values": "known hive identifier",
    },
    {
        "column": "inspection_timestamp",
        "description": ("Timestamp when the beekeeper inspected the hive."),
        "required": True,
        "allowed_values": "ISO 8601 date-time",
    },
    {
        "column": "harvest_decision",
        "description": ("Decision made before observing the eventual outcome."),
        "required": True,
        "allowed_values": ("harvest_now, wait, partial_harvest, no_harvest"),
    },
    {
        "column": "actual_harvest_timestamp",
        "description": "Confirmed time of completed harvest.",
        "required": False,
        "allowed_values": "ISO 8601 date-time",
    },
    {
        "column": "harvested_honey_mass_kg",
        "description": "Measured harvested honey mass.",
        "required": False,
        "allowed_values": "non-negative numeric",
    },
    {
        "column": "comb_capping_percent",
        "description": ("Estimated percentage of capped honey comb."),
        "required": False,
        "allowed_values": "0 to 100",
    },
    {
        "column": "honey_moisture_percent",
        "description": ("Measured honey moisture percentage."),
        "required": False,
        "allowed_values": "numeric percent",
    },
    {
        "column": "harvest_confirmed",
        "description": ("Whether the planned or observed harvest was completed."),
        "required": True,
        "allowed_values": "true, false",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the prospective beekeeper-confirmed harvest "
            "validation template and data dictionary."
        )
    )
    parser.add_argument(
        "--output-directory",
        default="data/prospective",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    backend_root = Path(__file__).resolve().parents[1]
    output_directory = backend_root / arguments.output_directory
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    template_path = output_directory / "harvest_validation_template.csv"
    dictionary_path = output_directory / "harvest_validation_data_dictionary.csv"

    pd.DataFrame(columns=TEMPLATE_COLUMNS).to_csv(
        template_path,
        index=False,
    )
    pd.DataFrame(DATA_DICTIONARY).to_csv(
        dictionary_path,
        index=False,
    )

    print("Created:", template_path)
    print("Created:", dictionary_path)


if __name__ == "__main__":
    main()
