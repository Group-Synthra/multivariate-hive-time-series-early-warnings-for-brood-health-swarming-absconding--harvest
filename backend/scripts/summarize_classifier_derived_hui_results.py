from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    report_root = backend_root / "artifacts/reports/harvesting/reviewed/classifier_derived_hui"

    distribution_path = report_root / "classifier_derived_hui_distribution.csv"
    summary_path = report_root / "future_hui_regression_summary.json"
    gate_path = report_root / "future_hui_regression_gate.json"

    print("\n" + "=" * 78)
    print("CURRENT CLASSIFIER-DERIVED HUI DISTRIBUTION")
    print("=" * 78)
    distribution = pd.read_csv(distribution_path)
    print(
        distribution.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("\n" + "=" * 78)
    print("FUTURE HUI REGRESSION")
    print("=" * 78)
    for horizon, result in summary["horizons"].items():
        validation = result["validation"]
        test = result["test"]
        print(f"\n{horizon}-HOUR HORIZON")
        print(f"Model: {result['selected_model']}")
        print(f"Feature set: {result['selected_feature_set']}")
        print(
            "Validation: "
            f"MAE={validation['mae']:.3f}, "
            f"RMSE={validation['rmse']:.3f}, "
            f"R2={validation['r2']:.3f}, "
            f"within ±5={validation['within_5_points_fraction'] * 100:.1f}%, "
            "class agreement="
            f"{validation['readiness_class_agreement_fraction'] * 100:.1f}%"
        )
        print(
            "Test:       "
            f"MAE={test['mae']:.3f}, "
            f"RMSE={test['rmse']:.3f}, "
            f"R2={test['r2']:.3f}, "
            f"within ±5={test['within_5_points_fraction'] * 100:.1f}%, "
            "class agreement="
            f"{test['readiness_class_agreement_fraction'] * 100:.1f}%"
        )

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    print("\n" + "=" * 78)
    print("RESEARCH GATE")
    print("=" * 78)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
