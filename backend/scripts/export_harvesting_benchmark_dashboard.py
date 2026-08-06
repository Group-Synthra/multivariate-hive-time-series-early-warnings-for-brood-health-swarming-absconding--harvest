from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            key: _json_safe(value)
            for key, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def _first_existing(
    columns: set[str],
    candidates: list[str],
) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        "None of the expected columns exist: "
        f"{candidates}"
    )


def _prepare_24h_series(
    predictions: pd.DataFrame,
    *,
    rows_per_hive: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    required = {"timestamp", "hive_id"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(
            "24-hour prediction file is missing columns: "
            f"{missing}"
        )

    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        errors="raise",
    )

    columns = set(frame.columns)
    current_weight = _first_existing(
        columns,
        [
            "robust_reference_weight_kg",
            "current_weight_kg",
            "weight_kg_current",
        ],
    )
    predicted_future_weight = _first_existing(
        columns,
        [
            "predicted_future_weight_kg",
            "predicted_robust_future_weight_kg",
        ],
    )
    actual_future_weight = _first_existing(
        columns,
        [
            "actual_future_weight_kg",
            "actual_robust_future_weight_kg",
        ],
    )
    predicted_delta = _first_existing(
        columns,
        [
            "predicted_delta_kg",
            "predicted_robust_delta_kg",
        ],
    )
    actual_delta = _first_existing(
        columns,
        [
            "actual_delta_kg",
            "actual_robust_delta_kg",
        ],
    )

    selected = frame[
        [
            "timestamp",
            "hive_id",
            current_weight,
            predicted_future_weight,
            actual_future_weight,
            predicted_delta,
            actual_delta,
        ]
    ].rename(
        columns={
            current_weight: "current_weight_kg",
            predicted_future_weight: (
                "predicted_future_weight_kg"
            ),
            actual_future_weight: (
                "actual_future_weight_kg"
            ),
            predicted_delta: "predicted_delta_kg",
            actual_delta: "actual_delta_kg",
        }
    )

    selected["absolute_error_kg"] = (
        selected["predicted_delta_kg"]
        - selected["actual_delta_kg"]
    ).abs()

    selected = (
        selected.sort_values(
            ["hive_id", "timestamp"]
        )
        .groupby("hive_id", group_keys=False)
        .tail(rows_per_hive)
        .reset_index(drop=True)
    )
    hives = sorted(
        selected["hive_id"].astype(str).unique().tolist()
    )
    return _records(selected), hives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export final harvesting benchmark status and "
            "exploratory 24-hour forecast data for the frontend."
        )
    )
    parser.add_argument(
        "--frontend-root",
        default=None,
    )
    parser.add_argument(
        "--rows-per-hive",
        type=int,
        default=168,
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

    report_root = (
        backend_root
        / "artifacts"
        / "reports"
        / "harvesting"
        / "reviewed"
    )

    decision = _read_json(
        report_root / "final_research_decision.json"
    )
    classification = _read_json(
        report_root
        / "research_models"
        / "selected_model_metrics.json"
    )
    alert_gate = _read_json(
        report_root
        / "alert_policy_gate"
        / "research_gate_summary.json"
    )
    robust_summary = _read_json(
        report_root
        / "robust_weight_forecasting"
        / "robust_weight_forecasting_summary.json"
    )
    robust_gate = _read_json(
        report_root
        / "robust_weight_forecasting"
        / "forecasting_research_gate.json"
    )

    predictions_path = (
        report_root
        / "robust_weight_forecasting"
        / "selected_test_predictions_24h.parquet"
    )
    predictions = pd.read_parquet(predictions_path)
    series, hives = _prepare_24h_series(
        predictions,
        rows_per_hive=arguments.rows_per_hive,
    )

    payload = {
        "decision": decision,
        "classification": classification,
        "alert_policy_gate": alert_gate,
        "robust_forecasting_summary": robust_summary,
        "robust_forecasting_gate": robust_gate,
        "exploratory_24h_series": series,
        "available_hives": hives,
        "display_rules": {
            "approved": [
                "Benchmark-only classifier",
                "Research-gate results",
                "Exploratory 24-hour weight forecast",
                "Prospective validation requirement",
            ],
            "blocked": [
                "HUI",
                "Harvest probability",
                "Validated readiness",
                "Recommended harvesting time",
            ],
        },
    }

    output_path = (
        frontend_root
        / "public"
        / "data"
        / "harvesting-research"
        / "benchmark-dashboard.json"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "exported",
                "output_path": str(output_path),
                "hive_count": len(hives),
                "forecast_rows": len(series),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
