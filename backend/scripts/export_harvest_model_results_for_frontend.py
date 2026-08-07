from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def export_model_dashboard(
    *,
    backend_root: Path,
    frontend_root: Path,
) -> dict[str, Any]:
    reports = (
        backend_root
        / "artifacts"
        / "reports"
        / "harvesting"
        / "reviewed"
        / "research_models"
    )

    selected_metrics = _read_json(
        reports / "selected_model_metrics.json"
    )
    comparison = _read_csv(
        reports / "model_feature_set_comparison.csv"
    )
    validation_events = _read_csv(
        reports / "selected_validation_event_detection.csv"
    )
    test_events = _read_csv(
        reports / "selected_test_event_detection.csv"
    )
    robustness = _read_csv(
        reports / "selected_grouped_hive_robustness.csv"
    )
    feature_importance = _read_csv(
        reports / "selected_feature_importance.csv"
    )
    session_summary = _read_csv(
        reports / "harvest_session_summary.csv"
    )

    successful = comparison.loc[
        comparison["status"].eq("ok")
    ].copy()

    sort_columns = [
        column
        for column in [
            "validation_event_recall",
            "validation_pr_auc",
            "validation_false_alert_episodes",
        ]
        if column in successful.columns
    ]
    if sort_columns:
        ascending = [
            column == "validation_false_alert_episodes"
            for column in sort_columns
        ]
        successful = successful.sort_values(
            sort_columns,
            ascending=ascending,
        )

    selected_model = selected_metrics["selected_model"]
    selected_feature_set = selected_metrics[
        "selected_feature_set"
    ]

    comparison["selected"] = (
        comparison["model"].eq(selected_model)
        & comparison["feature_set"].eq(selected_feature_set)
    )

    output_root = (
        frontend_root
        / "public"
        / "data"
        / "harvesting-models"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": selected_metrics,
        "candidate_count": len(comparison),
        "successful_candidate_count": len(successful),
        "comparison": _records(comparison),
        "validation_events": _records(validation_events),
        "test_events": _records(test_events),
        "grouped_hive_robustness": _records(robustness),
        "feature_importance": _records(
            feature_importance.head(25)
        ),
        "harvest_sessions": _records(session_summary),
        "display_notes": [
            (
                "PR-AUC is the primary row-level comparison metric "
                "because the target is strongly imbalanced."
            ),
            (
                "Validation contains only two reviewed events and "
                "test contains one reviewed event."
            ),
            (
                "The displayed scores are not yet calibrated HUI "
                "probabilities."
            ),
            (
                "The target represents probable harvest activity "
                "within 72 hours, not independently verified optimal "
                "honey maturity."
            ),
        ],
    }

    output_path = output_root / "dashboard.json"
    _write_json(output_path, payload)

    return {
        "output_path": str(output_path),
        "selected_model": selected_model,
        "selected_feature_set": selected_feature_set,
        "candidate_count": len(comparison),
        "successful_candidate_count": len(successful),
        "validation_event_count": len(validation_events),
        "test_event_count": len(test_events),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export reviewed harvesting model-comparison results "
            "for the Vite dashboard."
        )
    )
    parser.add_argument(
        "--frontend-root",
        default=None,
        help=(
            "Frontend directory. Defaults to the project-level "
            "frontend directory."
        ),
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
        raise FileNotFoundError(
            f"Frontend directory not found: {frontend_root}"
        )

    result = export_model_dashboard(
        backend_root=backend_root,
        frontend_root=frontend_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
