from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = BACKEND_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from multivari.modules.absconding.config import AbscondingSettings
from multivari.modules.absconding.data import run_absconding_data_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the separate historical Absconding dataset without changing the shared "
            "dataset used by the other BeeHive modules."
        )
    )
    parser.add_argument(
        "--config",
        default=str(BACKEND_ROOT / "config" / "absconding.yaml"),
        help="Absconding YAML configuration path.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=("CSV/XLSX path. Defaults to data.input_path in config/absconding.yaml."),
    )
    args = parser.parse_args()

    settings = AbscondingSettings.from_yaml(args.config)
    profile = run_absconding_data_pipeline(
        input_path=args.input or settings.data_input_path,
        backend_root=BACKEND_ROOT,
        config_path=args.config,
    )
    print("Separate Absconding data pipeline completed.")
    print(f"Rows: {profile.rows_clean:,}")
    print(f"Hives: {profile.hives}")
    print(f"Event onset markers: {profile.event_onset_markers}")
    print(f"Merged event episodes: {profile.merged_event_episodes}")
    print(f"24-hour positive rows: {profile.derived_24h_positive_rows:,}")
    print("Clean data: data/processed/absconding_clean.parquet")
    print("Split manifest: data/manifests/absconding_split_manifest.parquet")


if __name__ == "__main__":
    main()
