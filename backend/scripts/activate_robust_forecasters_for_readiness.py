from __future__ import annotations

from pathlib import Path

import yaml


def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    config_path = backend_root / "config/harvesting.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if "forecast_readiness" not in config:
        raise KeyError("forecast_readiness is missing from harvesting.yaml.")
    if "robust_weight_forecasting" not in config:
        raise KeyError(
            "robust_weight_forecasting is missing from harvesting.yaml."
        )

    robust = config["robust_weight_forecasting"]
    readiness = config["forecast_readiness"]
    readiness["comparison_path"] = (
        f"{robust['output_directory']}/"
        "robust_weight_forecasting_comparison.csv"
    )
    readiness["forecasting_summary_path"] = (
        f"{robust['output_directory']}/"
        "robust_weight_forecasting_summary.json"
    )
    readiness["forecaster_directory"] = robust["model_directory"]
    readiness["gate_output_path"] = robust["research_gate_path"]

    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print("Activated robust forecasters for readiness.")
    print("Readiness gate:", readiness["gate_output_path"])
    print("Forecaster directory:", readiness["forecaster_directory"])


if __name__ == "__main__":
    main()
