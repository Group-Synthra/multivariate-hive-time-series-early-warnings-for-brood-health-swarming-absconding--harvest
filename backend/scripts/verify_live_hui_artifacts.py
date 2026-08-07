from __future__ import annotations

import json
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "artifacts/models/harvesting/research_v2/selected_model.joblib",
    "artifacts/models/harvesting/research_v2/selected_feature_columns.json",
    "artifacts/models/harvesting/probability_calibration/selected_probability_calibrator.joblib",
    "artifacts/models/harvesting/classifier_derived_hui_regression/selected_classifier_derived_hui_regressor_24h.joblib",
    "artifacts/models/harvesting/classifier_derived_hui_regression/selected_classifier_derived_hui_regressor_24h.json",
    "artifacts/models/harvesting/classifier_derived_hui_regression/selected_classifier_derived_hui_regressor_48h.joblib",
    "artifacts/models/harvesting/classifier_derived_hui_regression/selected_classifier_derived_hui_regressor_48h.json",
    "artifacts/models/harvesting/classifier_derived_hui_regression/selected_classifier_derived_hui_regressor_72h.joblib",
    "artifacts/models/harvesting/classifier_derived_hui_regression/selected_classifier_derived_hui_regressor_72h.json",
    "artifacts/reports/harvesting/reviewed/probability_calibration/probability_calibration_gate.json",
    "artifacts/reports/harvesting/reviewed/classifier_derived_hui/future_hui_regression_gate.json",
    "config/harvesting.yaml",
    "data/processed/harvest_reviewed_feature_dataset.parquet",
    "src/multivari/modules/harvesting/classifier_derived_hui.py",
    "src/multivari/modules/harvesting/reviewed_features.py",
]


def main() -> None:
    records = []
    for relative in REQUIRED:
        path = BACKEND_ROOT / relative
        records.append(
            {
                "path": relative,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    missing = [record["path"] for record in records if not record["exists"]]
    print(
        json.dumps(
            {
                "status": "ok" if not missing else "missing_artifacts",
                "required_artifacts": records,
                "missing": missing,
            },
            indent=2,
        )
    )
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
