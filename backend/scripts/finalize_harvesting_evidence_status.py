from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the harvesting module using validation-gate, "
            "held-out generalization and robust-forecasting evidence."
        )
    )
    parser.add_argument(
        "--config",
        default="config/harvesting.yaml",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    backend_root = Path(__file__).resolve().parents[1]
    config_path = backend_root / arguments.config

    report_root = backend_root / "artifacts" / "reports" / "harvesting" / "reviewed"

    classification_review_path = (
        report_root / "alert_policy_gate" / "classification_generalization_review.json"
    )
    robust_gate_path = report_root / "robust_weight_forecasting" / "forecasting_research_gate.json"
    robust_summary_path = (
        report_root / "robust_weight_forecasting" / "robust_weight_forecasting_summary.json"
    )
    classification_metrics_path = report_root / "research_models" / "selected_model_metrics.json"

    classification_review = _read_json(classification_review_path)
    robust_gate = _read_json(robust_gate_path)
    robust_summary = _read_json(robust_summary_path)
    classification_metrics = _read_json(classification_metrics_path)

    classification_generalization = classification_review.get("generalization_supported") is True
    forecasting_ready = robust_gate.get("ready_for_readiness_prototype") is True

    if classification_generalization or forecasting_ready:
        research_status = "limited_evidence_review"
    else:
        research_status = "benchmark_only"

    horizon_24 = robust_gate["horizons"]["24"]
    horizon_48 = robust_gate["horizons"]["48"]
    horizon_72 = robust_gate["horizons"]["72"]

    decision = {
        "generated_at": datetime.now(UTC).isoformat(),
        "module": "Time-Optimal Honey Harvesting",
        "research_status": research_status,
        "operational_readiness_enabled": False,
        "calibration_enabled": False,
        "hui_enabled": False,
        "hrsi_enabled": False,
        "hrroc_enabled": False,
        "live_harvest_recommendation_enabled": False,
        "classification_branch": {
            "selected_model": classification_metrics.get("selected_model"),
            "selected_feature_set": (classification_metrics.get("selected_feature_set")),
            "validation_gate_passed": (classification_review["validation_gate_passed"]),
            "unchanged_test_event_supported": (
                classification_review["unchanged_test_event_supported"]
            ),
            "generalization_supported": (classification_review["generalization_supported"]),
            "validation_event_count": (classification_review["validation"]["event_count"]),
            "validation_detected_event_count": (
                classification_review["validation"]["detected_event_count"]
            ),
            "test_event_count": (classification_review["test"]["event_count"]),
            "test_detected_event_count": (classification_review["test"]["detected_event_count"]),
            "decision": classification_review["decision"],
        },
        "robust_forecasting_branch": {
            "gate_status": robust_gate["status"],
            "ready_for_readiness_prototype": (forecasting_ready),
            "improved_horizon_count": int(robust_gate["improved_horizon_count"]),
            "required_improved_horizons": int(robust_gate["required_improved_horizons"]),
            "decision": (
                "Do not construct a readiness score. Persistence remained best at 48 and 72 hours."
            ),
        },
        "exploratory_24h_forecast": {
            "allowed_for_research_display": bool(horizon_24["horizon_passed"]),
            "model": horizon_24["selected_model"],
            "feature_set": horizon_24["selected_feature_set"],
            "persistence_validation_mae": (horizon_24["persistence_validation_mae"]),
            "validation_mae": horizon_24["selected_validation_mae"],
            "test_mae": horizon_24["selected_test_mae"],
            "validation_mae_improvement_fraction": (
                horizon_24["validation_mae_improvement_fraction"]
            ),
            "restriction": (
                "Display only as an exploratory hive-weight "
                "forecast. It must not be translated into honey "
                "maturity or a harvest decision."
            ),
        },
        "longer_horizon_findings": {
            "48h_selected_model": horizon_48["selected_model"],
            "72h_selected_model": horizon_72["selected_model"],
            "48h_improvement_passed": bool(horizon_48["improvement_passed"]),
            "72h_improvement_passed": bool(horizon_72["improvement_passed"]),
        },
        "approved_dashboard_sections": [
            "Reviewed exploratory analysis",
            "Four-model classification benchmark",
            ("Validation alert-policy eligibility with held-out test miss"),
            "Robust forecasting comparison",
            "Exploratory 24-hour hive-weight forecast",
            "Prospective validation status",
        ],
        "blocked_dashboard_sections": [
            "Calibrated harvest probability",
            "HUI",
            "Validated readiness classes",
            "HRSI and HRRoC",
            "Recommended harvest window",
            "Live harvesting recommendation",
        ],
        "research_statement": (
            "The validation-selected alert policy detected both "
            "reviewed validation events but missed the unchanged "
            "held-out test event. Robust forecasting improved only "
            "the 24-hour horizon, while persistence remained best "
            "at 48 and 72 hours. The module is therefore retained "
            "as a benchmark and prospective-validation prototype."
        ),
        "next_research_stage": (
            "Prospectively collect beekeeper-confirmed harvest "
            "events, pre- and post-harvest weights, harvested honey "
            "mass, comb capping observations and honey moisture."
        ),
        "source_artifacts": {
            "classification_generalization_review": str(classification_review_path),
            "robust_forecasting_gate": str(robust_gate_path),
            "robust_forecasting_summary": str(robust_summary_path),
        },
        "robust_target_definition": robust_summary["target_definition"],
    }

    decision_path = report_root / "final_research_decision.json"
    _write_json(decision_path, decision)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["harvesting_research_status"] = {
        "status": research_status,
        "readiness_enabled": False,
        "calibration_enabled": False,
        "hui_enabled": False,
        "hrsi_enabled": False,
        "hrroc_enabled": False,
        "live_recommendation_enabled": False,
        "exploratory_24h_forecast_enabled": bool(horizon_24["horizon_passed"]),
        "classification_validation_gate_passed": (classification_review["validation_gate_passed"]),
        "classification_test_event_supported": (
            classification_review["unchanged_test_event_supported"]
        ),
        "decision_path": ("artifacts/reports/harvesting/reviewed/final_research_decision.json"),
    }
    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "finalized",
                "research_status": research_status,
                "operational_readiness_enabled": False,
                "classification_validation_gate_passed": (
                    classification_review["validation_gate_passed"]
                ),
                "classification_test_event_supported": (
                    classification_review["unchanged_test_event_supported"]
                ),
                "exploratory_24h_forecast_enabled": bool(horizon_24["horizon_passed"]),
                "decision_path": str(decision_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
