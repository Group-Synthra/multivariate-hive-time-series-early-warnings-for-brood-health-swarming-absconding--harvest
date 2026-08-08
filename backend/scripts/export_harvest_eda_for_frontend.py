from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _json_safe(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _split_balance_records(
    target_balance: pd.DataFrame,
    *,
    target_column: str,
) -> list[dict[str, Any]]:
    required = {"split", target_column, "rows"}
    missing = required.difference(target_balance.columns)
    if missing:
        raise ValueError(f"Target balance CSV is missing columns: {sorted(missing)}")

    records: list[dict[str, Any]] = []
    for split, group in target_balance.groupby("split", sort=False):
        by_target = {int(row[target_column]): int(row["rows"]) for _, row in group.iterrows()}
        negative = by_target.get(0, 0)
        positive = by_target.get(1, 0)
        total = negative + positive
        records.append(
            {
                "split": str(split),
                "negative_rows": negative,
                "positive_rows": positive,
                "total_rows": total,
                "positive_rate": positive / total if total else 0.0,
            }
        )
    return records


def export_dashboard_data(
    *,
    backend_root: Path,
    frontend_root: Path,
) -> dict[str, Any]:
    reports = backend_root / "artifacts" / "reports" / "harvesting" / "reviewed"

    target_audit = _read_json(reports / "target_audit.json")
    target_balance = _read_csv(reports / "target_balance_by_split.csv")
    feature_audit = _read_json(reports / "features" / "feature_audit.json")
    eda_audit = _read_json(reports / "feature_eda" / "reviewed_feature_eda_audit.json")
    grouped_summary = _read_json(
        reports / "grouped_hive_summary.json",
        required=False,
    )
    top_features = _read_csv(reports / "feature_eda" / "top_features_by_lead.csv")
    comparison = _read_csv(reports / "feature_eda" / "lead_feature_comparison.csv")
    coverage = _read_csv(reports / "feature_eda" / "sample_coverage.csv")

    target_column = "harvest_within_next_72h_reviewed"
    split_balance = _split_balance_records(
        target_balance,
        target_column=target_column,
    )

    output_root = frontend_root / "public" / "data" / "harvesting"
    figures_output = output_root / "figures"
    figures_output.mkdir(parents=True, exist_ok=True)

    source_figures = reports / "feature_eda" / "figures"
    figure_records: list[dict[str, Any]] = []
    if source_figures.exists():
        for figure in sorted(source_figures.glob("*.png")):
            destination = figures_output / figure.name
            shutil.copy2(figure, destination)
            lead_text = figure.stem.replace("top_features_lead_", "").replace("h", "")
            try:
                lead_hours = int(lead_text)
            except ValueError:
                lead_hours = None

            figure_records.append(
                {
                    "lead_hours": lead_hours,
                    "filename": figure.name,
                    "url": (f"/data/harvesting/figures/{figure.name}"),
                }
            )

    top_features_payload: dict[str, list[dict[str, Any]]] = {}
    if not top_features.empty:
        for lead, group in top_features.groupby(
            "lead_hours",
            sort=True,
        ):
            ordered = group.sort_values(
                "absolute_standardized_mean_difference",
                ascending=False,
            )
            top_features_payload[str(int(lead))] = _records(ordered)

    comparison_payload: dict[str, list[dict[str, Any]]] = {}
    if not comparison.empty:
        for lead, group in comparison.groupby(
            "lead_hours",
            sort=True,
        ):
            ordered = group.sort_values(
                "absolute_standardized_mean_difference",
                ascending=False,
            )
            comparison_payload[str(int(lead))] = _records(ordered)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "target": {
            "horizon_hours": int(target_audit["prediction_horizon_hours"]),
            "reviewed_event_count": int(target_audit["reviewed_event_count"]),
            "reviewed_positive_hives": int(target_audit["reviewed_positive_hives"]),
            "final_modelling_rows": int(target_audit["final_modelling_rows"]),
            "target_positive_rows": int(target_audit["target_positive_rows"]),
            "target_negative_rows": int(target_audit["target_negative_rows"]),
            "target_positive_rate": float(target_audit["target_positive_rate"]),
            "events_by_split": target_audit["reviewed_events_by_split"],
            "split_balance": split_balance,
        },
        "features": {
            "history_rows": feature_audit.get("history_rows"),
            "source_rows": int(feature_audit["source_rows"]),
            "output_rows": int(feature_audit["output_rows"]),
            "feature_count": int(feature_audit["feature_count"]),
            "minimum_history_hours": int(feature_audit["minimum_history_hours"]),
            "contiguous_segment_count": int(feature_audit["contiguous_segment_count"]),
            "detected_non_hourly_gaps": int(feature_audit["detected_non_hourly_gaps"]),
            "source_positive_rows": int(feature_audit["source_positive_rows"]),
            "output_positive_rows": int(feature_audit["output_positive_rows"]),
            "positive_rows_removed": int(feature_audit["positive_rows_removed"]),
            "output_positive_rate": float(feature_audit["output_positive_rate"]),
            "leakage_columns_present": feature_audit["leakage_columns_present"],
            "history_policy": feature_audit.get("history_policy"),
        },
        "eda": {
            "reviewed_event_count": int(eda_audit["reviewed_event_count"]),
            "expected_event_lead_samples": int(eda_audit["expected_event_lead_samples"]),
            "available_event_lead_samples": int(eda_audit["available_event_lead_samples"]),
            "available_matched_controls": int(eda_audit["available_matched_controls"]),
            "missing_event_lead_samples": int(eda_audit["missing_event_lead_samples"]),
            "missing_controls": int(eda_audit["missing_controls"]),
            "lead_hours": [int(value) for value in eda_audit["lead_hours"]],
            "control_exclusion_hours": int(eda_audit["control_exclusion_hours"]),
            "figure_count": int(eda_audit["figure_count"]),
            "figures": figure_records,
        },
        "grouped_validation": grouped_summary,
        "limitations": [
            (
                "The analysis uses 12 manually reviewed probable "
                "harvest events, not beekeeper-confirmed ground truth."
            ),
            (
                "Feature differences are exploratory and must not be "
                "interpreted as confirmatory statistical evidence."
            ),
            ("The official test split contains one event and should be reported as a case study."),
        ],
    }

    _write_json(output_root / "summary.json", summary)
    _write_json(
        output_root / "top-features.json",
        top_features_payload,
    )
    _write_json(
        output_root / "feature-comparison.json",
        comparison_payload,
    )
    _write_json(
        output_root / "sample-coverage.json",
        _records(coverage),
    )
    _write_json(
        output_root / "manifest.json",
        {
            "generated_at": summary["generated_at"],
            "files": [
                "summary.json",
                "top-features.json",
                "feature-comparison.json",
                "sample-coverage.json",
            ],
            "figure_count": len(figure_records),
        },
    )

    return {
        "output_directory": str(output_root),
        "reviewed_event_count": summary["target"]["reviewed_event_count"],
        "feature_count": summary["features"]["feature_count"],
        "lead_hours": summary["eda"]["lead_hours"],
        "figure_count": len(figure_records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Export reviewed harvesting EDA outputs as static JSON for the Vite frontend.")
    )
    parser.add_argument(
        "--frontend-root",
        default=None,
        help=("Frontend directory. Defaults to the project-level 'frontend' directory."),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    backend_root = Path(__file__).resolve().parents[1]
    project_root = backend_root.parent
    frontend_root = (
        Path(arguments.frontend_root).resolve()
        if arguments.frontend_root
        else project_root / "frontend"
    )

    if not frontend_root.exists():
        raise FileNotFoundError(f"Frontend directory not found: {frontend_root}")

    result = export_dashboard_data(
        backend_root=backend_root,
        frontend_root=frontend_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
