from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

HIVE_COLUMN = "hive_id"
TIMESTAMP_COLUMN = "timestamp"
SPLIT_COLUMN = "split"

REQUIRED_COMPONENTS = {
    "recent_accumulation",
    "weight_position",
    "forecast_plateau",
    "forecast_slowdown",
    "forecast_agreement",
    "environmental_stability",
}


def _resolve_path(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_safe),
        encoding="utf-8",
    )


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    frame_name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def evaluate_forecasting_research_gate(
    comparison: pd.DataFrame,
    summary: dict[str, Any],
    *,
    horizons_hours: list[int],
    minimum_validation_mae_improvement_fraction: float,
    required_improved_horizons: int,
    require_72h_not_worse_than_persistence: bool,
    maximum_test_to_validation_mae_ratio: float,
) -> dict[str, Any]:
    required_columns = {
        "horizon_hours",
        "model",
        "status",
        "validation_mae",
    }
    _require_columns(
        comparison,
        required_columns,
        frame_name="Weight forecasting comparison",
    )

    horizon_results: dict[str, Any] = {}
    improved_horizon_count = 0

    for horizon in horizons_hours:
        horizon_rows = comparison.loc[
            comparison["horizon_hours"].eq(horizon) & comparison["status"].eq("ok")
        ]
        persistence_rows = horizon_rows.loc[horizon_rows["model"].eq("persistence")]
        if persistence_rows.empty:
            raise ValueError(f"No successful persistence baseline exists at {horizon}h.")

        baseline_mae = float(persistence_rows["validation_mae"].min())
        selected = summary["horizons"][str(horizon)]
        selected_mae = float(selected["validation"]["mae"])
        test_mae = float(selected["test"]["mae"])

        improvement_fraction = (
            (baseline_mae - selected_mae) / baseline_mae if baseline_mae > 0 else 0.0
        )
        test_to_validation_ratio = test_mae / selected_mae if selected_mae > 0 else np.inf

        improvement_passed = improvement_fraction >= minimum_validation_mae_improvement_fraction
        generalization_passed = test_to_validation_ratio <= maximum_test_to_validation_mae_ratio
        horizon_passed = bool(improvement_passed and generalization_passed)
        improved_horizon_count += int(horizon_passed)

        horizon_results[str(horizon)] = {
            "selected_model": selected["selected_model"],
            "selected_feature_set": (selected["selected_feature_set"]),
            "persistence_validation_mae": baseline_mae,
            "selected_validation_mae": selected_mae,
            "selected_test_mae": test_mae,
            "validation_mae_improvement_fraction": (improvement_fraction),
            "test_to_validation_mae_ratio": (test_to_validation_ratio),
            "improvement_passed": improvement_passed,
            "generalization_passed": generalization_passed,
            "horizon_passed": horizon_passed,
        }

    horizon_72 = horizon_results.get("72")
    horizon_72_passed = True
    if require_72h_not_worse_than_persistence:
        if horizon_72 is None:
            raise ValueError("The 72-hour forecasting horizon is required.")
        horizon_72_passed = bool(
            horizon_72["selected_validation_mae"] <= horizon_72["persistence_validation_mae"]
            and horizon_72["generalization_passed"]
        )

    ready = bool(improved_horizon_count >= required_improved_horizons and horizon_72_passed)

    return {
        "status": ("forecasting_gate_passed" if ready else "forecasting_gate_failed"),
        "ready_for_readiness_prototype": ready,
        "minimum_validation_mae_improvement_fraction": (
            minimum_validation_mae_improvement_fraction
        ),
        "required_improved_horizons": (required_improved_horizons),
        "improved_horizon_count": improved_horizon_count,
        "require_72h_not_worse_than_persistence": (require_72h_not_worse_than_persistence),
        "horizon_72_passed": horizon_72_passed,
        "maximum_test_to_validation_mae_ratio": (maximum_test_to_validation_mae_ratio),
        "horizons": horizon_results,
        "interpretation": (
            "This gate checks whether future-weight forecasting "
            "adds measurable value beyond persistence before a "
            "readiness prototype is constructed."
        ),
    }


def run_forecasting_research_gate_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = config["forecast_readiness"]

    comparison_path = _resolve_path(
        root,
        settings["comparison_path"],
    )
    summary_path = _resolve_path(
        root,
        settings["forecasting_summary_path"],
    )
    gate_output_path = _resolve_path(
        root,
        settings["gate_output_path"],
    )

    comparison = pd.read_csv(comparison_path)
    summary = _read_json(summary_path)
    gate = settings["research_gate"]

    result = evaluate_forecasting_research_gate(
        comparison,
        summary,
        horizons_hours=[int(value) for value in settings["horizons_hours"]],
        minimum_validation_mae_improvement_fraction=float(
            gate["minimum_validation_mae_improvement_fraction"]
        ),
        required_improved_horizons=int(gate["required_improved_horizons"]),
        require_72h_not_worse_than_persistence=bool(gate["require_72h_not_worse_than_persistence"]),
        maximum_test_to_validation_mae_ratio=float(gate["maximum_test_to_validation_mae_ratio"]),
    )
    _write_json(gate_output_path, result)
    return result


def _robust_bounds(
    values: pd.Series,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float]:
    finite = values.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    if finite.empty:
        raise ValueError("Cannot estimate normalization bounds from empty values.")

    lower = float(finite.quantile(lower_quantile))
    upper = float(finite.quantile(upper_quantile))
    if upper <= lower:
        upper = lower + max(abs(lower) * 0.01, 1e-6)
    return lower, upper


def scale_higher_better(
    values: pd.Series,
    *,
    lower: float,
    upper: float,
) -> pd.Series:
    scaled = (values - lower) / (upper - lower)
    return scaled.clip(0.0, 1.0)


def scale_lower_better(
    values: pd.Series,
    *,
    lower: float,
    upper: float,
) -> pd.Series:
    return 1.0 - scale_higher_better(
        values,
        lower=lower,
        upper=upper,
    )


def add_contiguous_segment_id(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        frame,
        {HIVE_COLUMN, TIMESTAMP_COLUMN},
        frame_name="Readiness frame",
    )

    result = frame.copy()
    result[TIMESTAMP_COLUMN] = pd.to_datetime(
        result[TIMESTAMP_COLUMN],
        errors="raise",
    )
    result = result.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)

    elapsed = result.groupby(HIVE_COLUMN)[TIMESTAMP_COLUMN].diff().dt.total_seconds().div(3600)
    starts_segment = elapsed.isna() | elapsed.ne(1.0)
    result["_readiness_segment_id"] = starts_segment.cumsum().astype("int64")
    return result


def assign_readiness_class(
    score: pd.Series,
    thresholds: dict[str, float],
) -> pd.Series:
    approaching = float(thresholds["approaching"])
    ready = float(thresholds["ready"])
    high_priority = float(thresholds["high_priority"])

    if not approaching <= ready <= high_priority:
        raise ValueError("Readiness thresholds must be monotonically increasing.")

    labels = np.select(
        [
            score.ge(high_priority),
            score.ge(ready),
            score.ge(approaching),
        ],
        [
            "High Priority",
            "Ready",
            "Approaching",
        ],
        default="Not Ready",
    )
    return pd.Series(labels, index=score.index, dtype="string")


def assign_candidate_window(
    frame: pd.DataFrame,
    *,
    class_column: str,
    plateau_rate_thresholds: dict[str, float],
) -> pd.Series:
    _require_columns(
        frame,
        {
            class_column,
            "predicted_rate_24h_kg_per_hour",
            "predicted_rate_48h_kg_per_hour",
            "predicted_rate_72h_kg_per_hour",
        },
        frame_name="Readiness scores",
    )

    eligible = frame[class_column].isin(["Ready", "High Priority"])
    rate_24 = frame["predicted_rate_24h_kg_per_hour"].abs()
    rate_48 = frame["predicted_rate_48h_kg_per_hour"].abs()
    rate_72 = frame["predicted_rate_72h_kg_per_hour"].abs()

    labels = np.select(
        [
            eligible & rate_24.le(float(plateau_rate_thresholds["24"])),
            eligible & rate_48.le(float(plateau_rate_thresholds["48"])),
            eligible & rate_72.le(float(plateau_rate_thresholds["72"])),
        ],
        [
            "0-24 hours",
            "24-48 hours",
            "48-72 hours",
        ],
        default="No candidate window",
    )
    return pd.Series(labels, index=frame.index, dtype="string")


def _load_forecaster_predictions(
    feature_rows: pd.DataFrame,
    *,
    model_directory: Path,
    horizons_hours: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = feature_rows.copy()
    metadata: dict[str, Any] = {}

    for horizon in horizons_hours:
        model_path = model_directory / f"selected_weight_forecaster_{horizon}h.joblib"
        metadata_path = model_directory / f"selected_weight_forecaster_{horizon}h.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Selected forecaster not found: {model_path}")

        model_metadata = _read_json(metadata_path)
        feature_columns = [str(value) for value in model_metadata["feature_columns"]]
        _require_columns(
            result,
            set(feature_columns),
            frame_name=f"{horizon}h forecaster inputs",
        )

        estimator = joblib.load(model_path)
        prediction = estimator.predict(result[feature_columns])
        result[f"predicted_delta_{horizon}h_kg"] = np.asarray(prediction, dtype=float)
        result[f"predicted_rate_{horizon}h_kg_per_hour"] = (
            result[f"predicted_delta_{horizon}h_kg"] / horizon
        )
        metadata[str(horizon)] = model_metadata

    return result, metadata


def build_provisional_readiness_scores(
    feature_rows: pd.DataFrame,
    *,
    horizons_hours: list[int],
    component_weights: dict[str, float],
    lower_quantile: float,
    upper_quantile: float,
    plateau_rate_quantile: float,
    stability_window_hours: int,
    stability_minimum_periods: int,
    rate_of_change_hours: int,
    class_quantiles: dict[str, float],
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, float],
]:
    required_weights = set(component_weights)
    if required_weights != REQUIRED_COMPONENTS:
        raise ValueError(f"Component weights must contain exactly: {sorted(REQUIRED_COMPONENTS)}")

    weight_total = float(sum(component_weights.values()))
    if not np.isclose(weight_total, 1.0):
        raise ValueError("Readiness component weights must sum to one.")

    required_columns = {
        TIMESTAMP_COLUMN,
        HIVE_COLUMN,
        SPLIT_COLUMN,
        "weight_delta_72h_kg",
        "weight_distance_from_max_168h_kg",
        "temperature_c_std_24h",
        "co2_ppm_std_24h",
        "co2_flatline_24h_1",
        "co2_flatline_72h_1",
    }
    required_columns.update({f"predicted_delta_{horizon}h_kg" for horizon in horizons_hours})
    required_columns.update(
        {f"predicted_rate_{horizon}h_kg_per_hour" for horizon in horizons_hours}
    )
    _require_columns(
        feature_rows,
        required_columns,
        frame_name="Forecast-enriched feature rows",
    )

    frame = add_contiguous_segment_id(feature_rows)
    train_mask = frame[SPLIT_COLUMN].eq("train")
    if not train_mask.any():
        raise ValueError("Training rows are required for readiness normalization.")

    recent_accumulation = frame["weight_delta_72h_kg"].clip(lower=0.0)
    distance_from_max = frame["weight_distance_from_max_168h_kg"].clip(lower=0.0)

    rate_columns = [f"predicted_rate_{horizon}h_kg_per_hour" for horizon in horizons_hours]
    predicted_rates = frame[rate_columns]
    mean_absolute_future_rate = predicted_rates.abs().mean(axis=1)
    forecast_rate_std = predicted_rates.std(
        axis=1,
        ddof=0,
    )
    recent_rate = recent_accumulation / 72.0
    future_rate_72 = frame["predicted_rate_72h_kg_per_hour"].abs()
    forecast_slowdown = (recent_rate - future_rate_72).clip(lower=0.0)
    environmental_variability = frame["temperature_c_std_24h"] + frame["co2_ppm_std_24h"] / 100.0

    raw_components = {
        "recent_accumulation": recent_accumulation,
        "weight_position": distance_from_max,
        "forecast_plateau": mean_absolute_future_rate,
        "forecast_slowdown": forecast_slowdown,
        "forecast_agreement": forecast_rate_std,
        "environmental_stability": (environmental_variability),
    }

    parameters: dict[str, Any] = {
        "normalization_source_split": "train",
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "components": {},
    }

    component_scores: dict[str, pd.Series] = {}

    for name, values in raw_components.items():
        lower, upper = _robust_bounds(
            values.loc[train_mask],
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        parameters["components"][name] = {
            "lower": lower,
            "upper": upper,
            "direction": (
                "higher"
                if name
                in {
                    "recent_accumulation",
                    "forecast_slowdown",
                }
                else "lower"
            ),
        }

        if name in {
            "recent_accumulation",
            "forecast_slowdown",
        }:
            component_scores[name] = scale_higher_better(
                values,
                lower=lower,
                upper=upper,
            )
        else:
            component_scores[name] = scale_lower_better(
                values,
                lower=lower,
                upper=upper,
            )

    quality_penalty = (
        frame[
            [
                "co2_flatline_24h_1",
                "co2_flatline_72h_1",
            ]
        ]
        .max(axis=1)
        .clip(0.0, 1.0)
    )
    data_quality_score = 1.0 - 0.5 * quality_penalty

    weighted_score = pd.Series(
        0.0,
        index=frame.index,
        dtype="float64",
    )
    for name, weight in component_weights.items():
        frame[f"{name}_score"] = component_scores[name] * 100.0
        weighted_score += component_scores[name] * float(weight)

    frame["data_quality_score"] = data_quality_score * 100.0
    frame["provisional_readiness_score"] = (weighted_score * data_quality_score * 100.0).clip(
        0.0, 100.0
    )

    train_scores = frame.loc[
        train_mask,
        "provisional_readiness_score",
    ]
    thresholds = {
        name: float(train_scores.quantile(float(quantile)))
        for name, quantile in class_quantiles.items()
    }
    thresholds = {
        "approaching": thresholds["approaching"],
        "ready": max(
            thresholds["ready"],
            thresholds["approaching"],
        ),
        "high_priority": max(
            thresholds["high_priority"],
            thresholds["ready"],
        ),
    }

    frame["readiness_class"] = assign_readiness_class(
        frame["provisional_readiness_score"],
        thresholds,
    )

    grouped = frame.groupby(
        [HIVE_COLUMN, "_readiness_segment_id"],
        sort=False,
    )["provisional_readiness_score"]
    rolling_std = grouped.transform(
        lambda values: values.rolling(
            window=stability_window_hours,
            min_periods=stability_minimum_periods,
        ).std(ddof=0)
    )
    stability_scale = float(rolling_std.loc[train_mask].dropna().quantile(upper_quantile))
    if not np.isfinite(stability_scale) or stability_scale <= 0:
        stability_scale = 1.0

    frame["hrsi"] = (
        1.0 - (rolling_std.fillna(stability_scale) / stability_scale).clip(0.0, 1.0)
    ) * 100.0

    frame["hrroc_points_per_hour"] = (
        grouped.diff(periods=rate_of_change_hours) / rate_of_change_hours
    )

    plateau_thresholds = {
        str(horizon): float(
            frame.loc[
                train_mask,
                f"predicted_rate_{horizon}h_kg_per_hour",
            ]
            .abs()
            .quantile(plateau_rate_quantile)
        )
        for horizon in horizons_hours
    }
    frame["candidate_harvest_window"] = assign_candidate_window(
        frame,
        class_column="readiness_class",
        plateau_rate_thresholds=plateau_thresholds,
    )

    parameters["component_weights"] = component_weights
    parameters["stability_scale"] = stability_scale
    parameters["stability_window_hours"] = stability_window_hours
    parameters["rate_of_change_hours"] = rate_of_change_hours
    parameters["plateau_rate_thresholds"] = plateau_thresholds
    parameters["readiness_class_thresholds"] = thresholds

    output_columns = [column for column in frame.columns if not column.startswith("_")]
    return (
        frame[output_columns].copy(),
        parameters,
        thresholds,
    )


def run_provisional_readiness_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = config["forecast_readiness"]

    gate_path = _resolve_path(
        root,
        settings["gate_output_path"],
    )
    gate = _read_json(gate_path)
    if not bool(gate.get("ready_for_readiness_prototype", False)):
        raise RuntimeError(
            "The forecasting research gate did not pass. "
            "Do not build a readiness prototype from these forecasts."
        )

    feature_path = _resolve_path(
        root,
        settings["feature_dataset_path"],
    )
    model_directory = _resolve_path(
        root,
        settings["forecaster_directory"],
    )
    output_directory = _resolve_path(
        root,
        settings["output_directory"],
    )
    deployment_path = _resolve_path(
        root,
        settings["deployment_metadata_path"],
    )

    feature_rows = pd.read_parquet(feature_path)
    feature_rows[TIMESTAMP_COLUMN] = pd.to_datetime(
        feature_rows[TIMESTAMP_COLUMN],
        errors="raise",
    )
    horizons = [int(value) for value in settings["horizons_hours"]]

    enriched, forecaster_metadata = _load_forecaster_predictions(
        feature_rows,
        model_directory=model_directory,
        horizons_hours=horizons,
    )

    normalization = settings["normalization"]
    readiness, parameters, thresholds = build_provisional_readiness_scores(
        enriched,
        horizons_hours=horizons,
        component_weights={
            str(key): float(value) for key, value in settings["component_weights"].items()
        },
        lower_quantile=float(normalization["lower_quantile"]),
        upper_quantile=float(normalization["upper_quantile"]),
        plateau_rate_quantile=float(normalization["plateau_rate_quantile"]),
        stability_window_hours=int(settings["stability_window_hours"]),
        stability_minimum_periods=int(settings["stability_minimum_periods"]),
        rate_of_change_hours=int(settings["rate_of_change_hours"]),
        class_quantiles={
            str(key): float(value) for key, value in settings["readiness_class_quantiles"].items()
        },
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    deployment_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    score_path = output_directory / "provisional_readiness_scores.parquet"
    readiness.to_parquet(score_path, index=False)

    latest = (
        readiness.sort_values(TIMESTAMP_COLUMN)
        .groupby(HIVE_COLUMN, as_index=False)
        .tail(1)
        .sort_values(
            "provisional_readiness_score",
            ascending=False,
        )
    )
    latest_columns = [
        TIMESTAMP_COLUMN,
        HIVE_COLUMN,
        SPLIT_COLUMN,
        "provisional_readiness_score",
        "readiness_class",
        "hrsi",
        "hrroc_points_per_hour",
        "candidate_harvest_window",
        "predicted_delta_24h_kg",
        "predicted_delta_48h_kg",
        "predicted_delta_72h_kg",
        "data_quality_score",
    ]
    latest[latest_columns].to_csv(
        output_directory / "latest_provisional_readiness_by_hive.csv",
        index=False,
    )

    split_summary = (
        readiness.groupby(
            [SPLIT_COLUMN, "readiness_class"],
            observed=True,
        )
        .agg(
            rows=(
                "provisional_readiness_score",
                "size",
            ),
            mean_score=(
                "provisional_readiness_score",
                "mean",
            ),
            median_hrsi=("hrsi", "median"),
            mean_hrroc=(
                "hrroc_points_per_hour",
                "mean",
            ),
        )
        .reset_index()
    )
    split_summary.to_csv(
        output_directory / "readiness_distribution_by_split.csv",
        index=False,
    )

    _write_json(
        output_directory / "readiness_normalization_parameters.json",
        parameters,
    )
    _write_json(
        output_directory / "readiness_thresholds.json",
        {
            "status": "provisional_relative_thresholds",
            "thresholds": thresholds,
            "source": (
                "Quantiles of training-split provisional readiness "
                "scores; not beekeeper-validated action thresholds."
            ),
        },
    )

    summary = {
        "status": "provisional_readiness_prototype_built",
        "score_name": ("Provisional Harvest Readiness Score"),
        "score_range": [0, 100],
        "scored_rows": len(readiness),
        "hive_count": int(readiness[HIVE_COLUMN].nunique()),
        "forecasters": forecaster_metadata,
        "thresholds": thresholds,
        "class_counts": {
            str(key): int(value)
            for key, value in readiness["readiness_class"].value_counts().to_dict().items()
        },
        "research_position": (
            "This is a transparent relative readiness prototype "
            "derived from future-weight forecasts and current/past "
            "telemetry. It is not a calibrated harvest probability "
            "and does not verify honey maturity."
        ),
        "deployment_allowed": False,
        "next_required_stage": (
            "Retrospective case-study review, dashboard integration "
            "as a provisional score, and prospective beekeeper "
            "validation before operational recommendations."
        ),
    }
    _write_json(
        output_directory / "provisional_readiness_summary.json",
        summary,
    )
    _write_json(
        deployment_path,
        {
            "deployment_allowed": False,
            "score_name": summary["score_name"],
            "score_status": "provisional_relative_score",
            "thresholds": thresholds,
            "normalization_parameters_path": str(
                output_directory / "readiness_normalization_parameters.json"
            ),
            "warning": summary["research_position"],
        },
    )

    return {
        "status": summary["status"],
        "scored_rows": len(readiness),
        "hive_count": summary["hive_count"],
        "score_path": str(score_path),
        "summary_path": str(output_directory / "provisional_readiness_summary.json"),
        "latest_path": str(output_directory / "latest_provisional_readiness_by_hive.csv"),
    }
