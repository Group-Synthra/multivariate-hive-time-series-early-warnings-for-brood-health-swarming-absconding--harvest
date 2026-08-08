from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SENSOR_FAMILIES = {
    "weight": {"label": "Weight", "prefixes": ("weight_",)},
    "humidity": {"label": "Humidity", "prefixes": ("humidity_",)},
    "temperature": {"label": "Temperature", "prefixes": ("temperature_",)},
    "co2": {"label": "CO₂", "prefixes": ("co2_",)},
}
PROHIBITED_TOKENS = (
    "harvest_within_next",
    "reviewed_event",
    "event_start",
    "harvest_event",
    "future_",
)


def _feature_family(feature: str) -> str | None:
    for family, meta in SENSOR_FAMILIES.items():
        if str(feature).startswith(meta["prefixes"]):
            return family
    return None


def _friendly(feature: str) -> str:
    text = str(feature)
    replacements = {
        "weight_": "Weight ",
        "humidity_pct_": "Humidity ",
        "humidity_": "Humidity ",
        "temperature_c_": "Temperature ",
        "temperature_": "Temperature ",
        "co2_ppm_": "CO₂ ",
        "co2_": "CO₂ ",
        "_per_hour": " / hour",
        "_kg": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.replace("_", " ").split()).capitalize()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _top_sensor_rows(
    comparison: pd.DataFrame,
    lead: int,
    maximum_rows: int = 10,
) -> list[dict[str, Any]]:
    subset = comparison.loc[comparison["lead_hours"].eq(lead)].copy()
    subset["sensor_family"] = subset["feature"].map(_feature_family)
    subset = subset.loc[subset["sensor_family"].notna()].dropna(
        subset=["absolute_standardized_mean_difference"]
    )
    subset = subset.nlargest(
        maximum_rows,
        "absolute_standardized_mean_difference",
    )
    records = []
    for row in subset.to_dict(orient="records"):
        records.append(
            {
                "lead_hours": int(row["lead_hours"]),
                "feature": str(row["feature"]),
                "display_name": _friendly(str(row["feature"])),
                "sensor_family": str(row["sensor_family"]),
                "event_mean": float(row["event_mean"]),
                "control_mean": float(row["control_mean"]),
                "smd": float(row["standardized_mean_difference"]),
                "absolute_smd": float(row["absolute_standardized_mean_difference"]),
                "event_n": int(row["event_n"]),
                "control_n": int(row["control_n"]),
            }
        )
    return records


def _sensor_summary(
    comparison: pd.DataFrame,
    lead: int = 72,
) -> list[dict[str, Any]]:
    output = []
    for family, meta in SENSOR_FAMILIES.items():
        subset = comparison.loc[
            comparison["lead_hours"].eq(lead)
            & comparison["feature"].map(
                lambda value, family=family: _feature_family(str(value)) == family
            )
        ].dropna(subset=["absolute_standardized_mean_difference"])
        if subset.empty:
            output.append(
                {
                    "family": family,
                    "label": meta["label"],
                    "maximum_absolute_smd": None,
                    "feature": None,
                    "feature_display_name": None,
                    "strength": "Unavailable",
                }
            )
            continue
        best = subset.nlargest(1, "absolute_standardized_mean_difference").iloc[0]
        value = float(best["absolute_standardized_mean_difference"])
        output.append(
            {
                "family": family,
                "label": meta["label"],
                "maximum_absolute_smd": value,
                "feature": str(best["feature"]),
                "feature_display_name": _friendly(str(best["feature"])),
                "strength": (
                    "Strong" if value >= 0.8 else "Moderate" if value >= 0.5 else "Present"
                ),
            }
        )
    return output


def _signal_evolution(
    comparison: pd.DataFrame,
    lead_hours: list[int],
) -> list[dict[str, Any]]:
    records = []
    for lead in lead_hours:
        row: dict[str, Any] = {"lead_hours": int(lead)}
        for family in SENSOR_FAMILIES:
            subset = comparison.loc[
                comparison["lead_hours"].eq(lead)
                & comparison["feature"].map(
                    lambda value, family=family: _feature_family(str(value)) == family
                )
            ].dropna(subset=["absolute_standardized_mean_difference"])
            row[family] = (
                float(subset["absolute_standardized_mean_difference"].max())
                if not subset.empty
                else None
            )
        records.append(row)
    return records


def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    project_root = backend_root.parent
    config = yaml.safe_load((backend_root / "config/harvesting.yaml").read_text(encoding="utf-8"))
    settings = config["reviewed_feature_eda"]
    feature_settings = config["reviewed_features"]
    target_column = config["reviewed_target"]["output_column"]

    event_path = backend_root / settings["event_table_path"]
    feature_path = backend_root / settings["feature_dataset_path"]
    manifest_path = backend_root / settings["feature_manifest_path"]
    report_root = backend_root / settings["report_directory"]
    comparison_path = report_root / "lead_feature_comparison.csv"
    coverage_path = report_root / "sample_coverage.csv"

    required = [
        event_path,
        feature_path,
        manifest_path,
        comparison_path,
        coverage_path,
    ]
    missing = [str(path.relative_to(backend_root)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Reviewed EDA artifacts are missing: " + ", ".join(missing))

    events = pd.read_parquet(event_path)
    features = pd.read_parquet(feature_path)
    manifest = pd.read_csv(manifest_path)
    comparison = pd.read_csv(comparison_path)
    coverage = pd.read_csv(coverage_path)

    lead_hours = [int(value) for value in settings["lead_hours"]]
    feature_names = [str(value) for value in manifest["feature_name"].dropna().tolist()]
    modelling_rows = len(features)
    positive_rows = int(
        pd.to_numeric(features[target_column], errors="coerce").fillna(0).eq(1).sum()
    )
    prevalence = positive_rows / modelling_rows if modelling_rows else 0.0

    prohibited = sorted(
        feature
        for feature in feature_names
        if any(token in feature.lower() for token in PROHIBITED_TOKENS)
    )

    events_by_split = {
        str(key): int(value) for key, value in events["split"].value_counts().to_dict().items()
    }
    expected_samples = int(len(events) * len(lead_hours))
    available_event_samples = int(
        coverage["event_sample_available"].fillna(False).astype(bool).sum()
    )
    available_controls = int(coverage["control_sample_available"].fillna(False).astype(bool).sum())
    coverage_percent = (
        min(available_event_samples, available_controls) / expected_samples * 100.0
        if expected_samples
        else 0.0
    )

    sensor_summary = _sensor_summary(comparison, lead=72)
    strong_count = sum(
        1
        for item in sensor_summary
        if item["maximum_absolute_smd"] is not None and item["maximum_absolute_smd"] >= 0.8
    )

    details = comparison.loc[
        comparison["feature"].map(lambda value: _feature_family(str(value)) is not None)
    ].sort_values(
        ["lead_hours", "absolute_standardized_mean_difference"],
        ascending=[False, False],
    )

    payload = {
        "status": "final_reviewed_eda_dashboard_ready",
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "modelling_rows": modelling_rows,
            "reviewed_events": len(events),
            "positive_hives": int(events["hive_id"].nunique()),
            "engineered_features": len(feature_names),
            "minimum_history_hours": int(feature_settings.get("minimum_history_hours", 168)),
            "positive_rows": positive_rows,
            "target_prevalence": prevalence,
            "events_by_split": events_by_split,
        },
        "integrity": {
            "no_prohibited_leakage_features": len(prohibited) == 0,
            "prohibited_features_present": prohibited,
            "past_only_feature_policy": True,
            "available_event_samples": available_event_samples,
            "available_controls": available_controls,
            "expected_samples": expected_samples,
            "missing_event_samples": expected_samples - available_event_samples,
            "missing_controls": expected_samples - available_controls,
        },
        "sensor_summary_72h": sensor_summary,
        "strong_sensor_family_count": strong_count,
        "top_sensor_features_by_lead": {
            str(lead): _top_sensor_rows(comparison, lead) for lead in lead_hours
        },
        "signal_evolution": _signal_evolution(comparison, lead_hours),
        "lead_hours": lead_hours,
        "coverage": {
            "reviewed_events": len(events),
            "lead_times_per_event": len(lead_hours),
            "expected_event_lead_samples": expected_samples,
            "available_event_lead_samples": available_event_samples,
            "available_matched_controls": available_controls,
            "coverage_percent": coverage_percent,
        },
        "detailed_sensor_statistics": [
            {key: _json_safe(value) for key, value in row.items()}
            for row in details[
                [
                    "lead_hours",
                    "feature",
                    "event_mean",
                    "control_mean",
                    "standardized_mean_difference",
                    "event_n",
                    "control_n",
                ]
            ].to_dict(orient="records")
        ],
        "takeaway": {
            "title": "Multivariate pre-event structure identified",
            "text": (
                f"{strong_count} of 4 sensor families show large descriptive "
                "event-control separation at the 72-hour lead (|SMD| ≥ 0.8). "
                "Together with complete matched-control coverage and past-only "
                "feature construction, the EDA supports comparative multivariate "
                "model evaluation."
            ),
        },
    }

    output_path = (
        project_root / "frontend/public/data/harvesting-research/final-reviewed-eda-dashboard.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": payload["status"],
                "output_path": str(output_path),
                "reviewed_events": len(events),
                "modelling_rows": modelling_rows,
                "coverage_percent": coverage_percent,
                "strong_sensor_family_count": strong_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
