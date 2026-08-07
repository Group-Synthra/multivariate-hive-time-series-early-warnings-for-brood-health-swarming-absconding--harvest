from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    report_root = (
        backend_root
        / "artifacts"
        / "reports"
        / "harvesting"
        / "reviewed"
        / "research_models"
    )

    comparison_path = (
        report_root / "model_feature_set_comparison.csv"
    )
    predictions_path = (
        report_root / "selected_validation_predictions.parquet"
    )

    comparison = pd.read_csv(comparison_path)
    predictions = pd.read_parquet(predictions_path)

    target_column = "harvest_within_next_72h_reviewed"
    probability_column = "raw_probability"

    prevalence = float(predictions[target_column].mean())
    probability = predictions[probability_column]

    successful = comparison.loc[
        comparison["status"].eq("ok")
    ].copy()
    successful["validation_no_skill_pr_auc"] = prevalence
    successful["validation_pr_auc_lift"] = (
        successful["validation_pr_auc"] / prevalence
    )

    columns = [
        "model",
        "feature_set",
        "validation_pr_auc",
        "validation_no_skill_pr_auc",
        "validation_pr_auc_lift",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_event_recall",
        "validation_false_alert_episodes",
    ]

    print("\nVALIDATION PREVALENCE / NO-SKILL PR-AUC")
    print(f"{prevalence:.10f}")

    print("\nSELECTED PROBABILITY DISTRIBUTION")
    print(
        probability.describe(
            percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]
        ).to_string()
    )
    print(
        "\nUnique probabilities rounded to 8 decimals:",
        probability.round(8).nunique(),
    )

    print("\nMODEL COMPARISON WITH PR-AUC LIFT")
    print(
        successful[columns]
        .sort_values(
            "validation_pr_auc_lift",
            ascending=False,
        )
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    best_lift = float(
        successful["validation_pr_auc_lift"].max()
    )
    probability_std = float(probability.std())

    print("\nDIAGNOSTIC")
    if probability_std < 1e-6:
        print(
            "FAIL: Selected probabilities are effectively constant."
        )
    elif best_lift <= 1.05:
        print(
            "FAIL: No candidate materially exceeds the no-skill "
            "PR-AUC baseline."
        )
    else:
        print(
            "PASS: At least one candidate exceeds the no-skill "
            "baseline. Continue with event-level and robustness review."
        )


if __name__ == "__main__":
    main()
