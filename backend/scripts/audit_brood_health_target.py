from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from multivari.modules.brood_health.audit import binary_target_persistence_audit
from multivari.modules.brood_health.config import PATHS
from multivari.modules.brood_health.features import normalise_historical
from multivari.modules.brood_health.scoring import score_definition


def load_source(path: Path | None) -> pd.DataFrame:
    source = Path(path or PATHS.clean_data)
    if source.exists():
        suffix = source.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return normalise_historical(pd.read_excel(source, sheet_name="Common_Dataset"))
        if suffix == ".csv":
            return normalise_historical(pd.read_csv(source))
        try:
            return normalise_historical(pd.read_parquet(source))
        except ImportError:
            pass
    if PATHS.raw_workbook.exists():
        return normalise_historical(pd.read_excel(PATHS.raw_workbook, sheet_name="Common_Dataset"))
    raise FileNotFoundError("No common cleaned dataset or raw workbook was found")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit binary brood-status persistence before training the score forecast model."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24])
    args = parser.parse_args()

    frame = load_source(args.input)
    payload = {
        "binary_target_persistence": binary_target_persistence_audit(
            frame,
            horizons=args.horizons,
        ),
        "corrected_primary_target": {
            "name": "future_minimum_brood_health_score",
            "description": (
                "Minimum sensor-derived Brood Health Score observed inside the future forecast window."
            ),
            "score_definition": score_definition(),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
