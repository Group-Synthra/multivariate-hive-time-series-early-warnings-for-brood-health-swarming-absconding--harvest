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
            "Freeze the harvesting module as a benchmark-only "
            "research result after both operational gates."
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

    report_root = (
        backend_root
        / "artifacts"
        / "reports"
        / "harvesting"
        / "reviewed"
    )

    alert_gate_path = (
        report_root
        / "alert_policy_gate"
        / "research_gate_summary.json"
    )
    robust_gate_path = (
        report_root
        / "robust_weight_forecasting"
        / "forecasting_research_gate.json"
    )
    robust_summary_path = (
        report_root
        / "robust_weight_forecasting"
        / "robust_weight_forecasting_summary.json"
    )
    classification_path = (
        report_root
        / "research_models"
        / "selected_model_metrics.json"
    )

    alert_gate = _read_json(alert_gate_path)
    robust_gate = _read_json(robust_gate_path)
    robust_summary = _read_json(robust_summary_path)
    classification = _read_json(classification_path)

    classification_ready = (
        alert_gate.get("ready_for_calibration") is True
    )
    forecasting_ready = (
        robust_gate.get(
            "ready_for_readiness_prototype"
        )
        is True
    )

    if classification_ready or forecasting_ready:
        raise RuntimeError(
            "At least one research gate passed. This script is only "
            "for the current benchmark-only outcome."
        )

    horizon_24 = robust_gate["horizons"]["24"]
    horizon_48 = robust_gate["horizons"]["48"]
    horizon_72 = robust_gate["horizons"]["72"]

    decision = {
        "generated_at": datetime.now(UTC).isoformat(),
        "module": "Time-Optimal Honey Harvesting",
        "research_status": "benchmark_only",
        "operational_readiness_enabled": False,
        "calibration_enabled": False,
        "hui_enabled": False,
        "hrsi_enabled": False,
        "hrroc_enabled": False,
        "live_harvest_recommendation_enabled": False,
        "classification_branch": {
            "gate_status": alert_gate["status"],
            "ready_for_calibration": classification_ready,
            "selected_model": classification.get(
                "selected_model"
            ),
            "selected_feature_set": classification.get(
                "selected_feature_set"
            ),
            "decision": (
                "Retain the probable-harvest classifier as a "
                "benchmark experiment. No research-safe temporal "
                "alert policy was identified."
            ),
        },
        "robust_forecasting_branch": {
            "gate_status": robust_gate["status"],
            "ready_for_readiness_prototype": forecasting_ready,
            "improved_horizon_count": int(
                robust_gate["improved_horizon_count"]
            ),
            "required_improved_horizons": int(
                robust_gate["required_improved_horizons"]
            ),
            "horizons": robust_gate["horizons"],
            "decision": (
                "Do not construct a harvest-readiness score. "
                "Persistence remained best at 48 and 72 hours."
            ),
        },
        "exploratory_24h_forecast": {
            "allowed_for_research_display": bool(
                horizon_24["horizon_passed"]
            ),
            "model": horizon_24["selected_model"],
            "feature_set": horizon_24[
                "selected_feature_set"
            ],
            "persistence_validation_mae": horizon_24[
                "persistence_validation_mae"
            ],
            "validation_mae": horizon_24[
                "selected_validation_mae"
            ],
            "test_mae": horizon_24[
                "selected_test_mae"
            ],
            "validation_mae_improvement_fraction": (
                horizon_24[
                    "validation_mae_improvement_fraction"
                ]
            ),
            "test_to_validation_mae_ratio": horizon_24[
                "test_to_validation_mae_ratio"
            ],
            "restriction": (
                "Display only as an exploratory hive-weight "
                "forecast. It must not be translated into harvest "
                "readiness, honey maturity or a harvest decision."
            ),
        },
        "longer_horizon_findings": {
            "48h_selected_model": horizon_48[
                "selected_model"
            ],
            "72h_selected_model": horizon_72[
                "selected_model"
            ],
            "48h_improvement_passed": bool(
                horizon_48["improvement_passed"]
            ),
            "72h_improvement_passed": bool(
                horizon_72["improvement_passed"]
            ),
        },
        "approved_dashboard_sections": [
            "Reviewed exploratory analysis",
            "Four-model classification benchmark",
            "Alert-policy gate failure",
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
        "next_research_stage": (
            "Prospectively collect beekeeper-confirmed harvest "
            "events, pre- and post-harvest weights, harvested honey "
            "mass, comb capping observations and honey moisture."
        ),
        "research_statement": (
            "Neither the event classifier nor the 48- and 72-hour "
            "future-weight forecasters demonstrated sufficient "
            "evidence for an operational harvest-readiness score. "
            "The module is therefore reported as a benchmark and "
            "prospective-validation prototype."
        ),
        "source_artifacts": {
            "alert_gate": str(alert_gate_path),
            "robust_forecasting_gate": str(
                robust_gate_path
            ),
            "robust_forecasting_summary": str(
                robust_summary_path
            ),
        },
        "robust_target_definition": robust_summary[
            "target_definition"
        ],
    }

    decision_path = (
        report_root / "final_research_decision.json"
    )
    _write_json(decision_path, decision)

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )
    config["harvesting_research_status"] = {
        "status": "benchmark_only",
        "readiness_enabled": False,
        "calibration_enabled": False,
        "hui_enabled": False,
        "hrsi_enabled": False,
        "hrroc_enabled": False,
        "live_recommendation_enabled": False,
        "exploratory_24h_forecast_enabled": bool(
            horizon_24["horizon_passed"]
        ),
        "decision_path": (
            "artifacts/reports/harvesting/reviewed/"
            "final_research_decision.json"
        ),
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
                "research_status": "benchmark_only",
                "operational_readiness_enabled": False,
                "exploratory_24h_forecast_enabled": bool(
                    horizon_24["horizon_passed"]
                ),
                "decision_path": str(decision_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
