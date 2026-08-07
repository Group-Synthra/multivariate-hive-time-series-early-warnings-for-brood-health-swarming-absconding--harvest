from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TIMESTAMP_COLUMN = "timestamp"
HIVE_COLUMN = "hive_id"
SPLIT_COLUMN = "split"
TARGET_COLUMN = "harvest_within_next_72h_reviewed"
RAW_PROBABILITY_COLUMN = "raw_probability"
CALIBRATED_PROBABILITY_COLUMN = "calibrated_probability"
CURRENT_HUI_COLUMN = "classifier_derived_hui"
CURRENT_CLASS_COLUMN = "harvest_readiness_class"
SEGMENT_COLUMN = "_hui_segment_id"

HUI_HISTORY_COLUMNS = [
    "hui_delta_1h",
    "hui_delta_6h",
    "hui_delta_24h",
    "hui_mean_6h",
    "hui_std_6h",
    "hui_mean_24h",
    "hui_std_24h",
    "hui_trend_6h_per_hour",
    "hui_trend_24h_per_hour",
]

NON_FEATURE_COLUMNS = {
    TIMESTAMP_COLUMN,
    HIVE_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    RAW_PROBABILITY_COLUMN,
    CALIBRATED_PROBABILITY_COLUMN,
    CURRENT_CLASS_COLUMN,
}

MODEL_COMPLEXITY = {
    "persistence": 0,
    "ridge": 1,
    "random_forest": 2,
    "xgboost": 3,
    "lightgbm": 4,
}


@dataclass(frozen=True)
class CandidateEvaluation:
    horizon_hours: int
    model_name: str
    feature_set_name: str
    feature_columns: list[str]
    fitted_estimator: Any
    validation_prediction: np.ndarray
    test_prediction: np.ndarray
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]


class PersistenceHuiRegressor:
    """Predict that future HUI remains equal to the current HUI."""

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> PersistenceHuiRegressor:
        del target
        if CURRENT_HUI_COLUMN not in features.columns:
            raise ValueError(
                f"{CURRENT_HUI_COLUMN} is required for persistence."
            )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (
            pd.to_numeric(
                features[CURRENT_HUI_COLUMN],
                errors="raise",
            )
            .to_numpy(dtype=float)
            .clip(0.0, 100.0)
        )


def _resolve_path(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
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
        raise ValueError(
            f"{frame_name} is missing required columns: {missing}"
        )


def _validate_anchors(
    probability_anchors: list[float],
    hui_anchors: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    if len(probability_anchors) != len(hui_anchors):
        raise ValueError(
            "Probability and HUI anchor lists must have equal length."
        )
    if len(probability_anchors) < 2:
        raise ValueError("At least two HUI anchors are required.")

    probabilities = np.asarray(probability_anchors, dtype=float)
    hui_values = np.asarray(hui_anchors, dtype=float)

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probability anchors must be finite.")
    if not np.all(np.isfinite(hui_values)):
        raise ValueError("HUI anchors must be finite.")
    if not np.all(np.diff(probabilities) > 0):
        raise ValueError(
            "Probability anchors must be strictly increasing."
        )
    if not np.all(np.diff(hui_values) > 0):
        raise ValueError("HUI anchors must be strictly increasing.")
    if probabilities[0] < 0.0 or probabilities[-1] > 1.0:
        raise ValueError("Probability anchors must lie inside 0–1.")
    if hui_values[0] < 0.0 or hui_values[-1] > 100.0:
        raise ValueError("HUI anchors must lie inside 0–100.")

    return probabilities, hui_values


def probability_to_hui(
    values: pd.Series | np.ndarray,
    *,
    probability_anchors: list[float],
    hui_anchors: list[float],
) -> np.ndarray:
    """Map adjusted classifier scores to a monotonic 0–100 HUI."""
    probabilities, hui_values = _validate_anchors(
        probability_anchors,
        hui_anchors,
    )
    input_values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(input_values)):
        raise ValueError("Classifier scores must be finite.")

    return np.interp(
        input_values,
        probabilities,
        hui_values,
        left=hui_values[0],
        right=hui_values[-1],
    ).clip(0.0, 100.0)


def assign_harvest_readiness_class(
    values: pd.Series | np.ndarray,
    *,
    not_ready_upper: float,
    approaching_upper: float,
    ready_upper: float,
) -> pd.Series:
    if not (
        0.0
        < not_ready_upper
        < approaching_upper
        < ready_upper
        < 100.0
    ):
        raise ValueError(
            "HUI class boundaries must increase inside 0–100."
        )

    series = pd.Series(values, copy=False, dtype=float)
    labels = np.select(
        [
            series.lt(not_ready_upper),
            series.lt(approaching_upper),
            series.lt(ready_upper),
        ],
        [
            "Not Ready",
            "Approaching Harvest",
            "Ready",
        ],
        default="High-Priority Harvest",
    )
    return pd.Series(labels, index=series.index, dtype="string")


def _add_contiguous_segment_id(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(
        [HIVE_COLUMN, TIMESTAMP_COLUMN]
    ).reset_index(drop=True)
    elapsed_hours = (
        result.groupby(HIVE_COLUMN)[TIMESTAMP_COLUMN]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )
    previous_split = result.groupby(HIVE_COLUMN)[SPLIT_COLUMN].shift()
    starts_segment = (
        elapsed_hours.isna()
        | elapsed_hours.ne(1.0)
        | result[SPLIT_COLUMN].ne(previous_split)
    )
    result[SEGMENT_COLUMN] = starts_segment.cumsum().astype("int64")
    return result


def add_hui_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        frame,
        {
            TIMESTAMP_COLUMN,
            HIVE_COLUMN,
            SPLIT_COLUMN,
            CURRENT_HUI_COLUMN,
        },
        frame_name="Classifier-derived HUI rows",
    )

    result = frame.copy()
    result[TIMESTAMP_COLUMN] = pd.to_datetime(
        result[TIMESTAMP_COLUMN],
        errors="raise",
    )
    result = _add_contiguous_segment_id(result)
    grouped = result.groupby(SEGMENT_COLUMN, sort=False)[
        CURRENT_HUI_COLUMN
    ]

    for lag in (1, 6, 24):
        result[f"hui_delta_{lag}h"] = (
            result[CURRENT_HUI_COLUMN] - grouped.shift(lag)
        )

    for window in (6, 24):
        rolling = grouped.rolling(
            window=window,
            min_periods=window,
        )
        result[f"hui_mean_{window}h"] = (
            rolling.mean().reset_index(level=0, drop=True)
        )
        result[f"hui_std_{window}h"] = (
            rolling.std(ddof=0).reset_index(level=0, drop=True)
        )

    result["hui_trend_6h_per_hour"] = result["hui_delta_6h"] / 6.0
    result["hui_trend_24h_per_hour"] = (
        result["hui_delta_24h"] / 24.0
    )
    return result.drop(columns=[SEGMENT_COLUMN])


def add_future_hui_target(
    frame: pd.DataFrame,
    *,
    horizon_hours: int,
) -> pd.DataFrame:
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive.")

    _require_columns(
        frame,
        {
            TIMESTAMP_COLUMN,
            HIVE_COLUMN,
            SPLIT_COLUMN,
            CURRENT_HUI_COLUMN,
        },
        frame_name="Classifier-derived HUI rows",
    )

    current = frame.copy()
    current[TIMESTAMP_COLUMN] = pd.to_datetime(
        current[TIMESTAMP_COLUMN],
        errors="raise",
    )
    current = _add_contiguous_segment_id(current)
    current["_future_timestamp"] = (
        current[TIMESTAMP_COLUMN]
        + pd.to_timedelta(horizon_hours, unit="h")
    )

    target_column = f"future_classifier_derived_hui_{horizon_hours}h"
    future = current[
        [
            HIVE_COLUMN,
            SPLIT_COLUMN,
            SEGMENT_COLUMN,
            TIMESTAMP_COLUMN,
            CURRENT_HUI_COLUMN,
        ]
    ].rename(
        columns={
            TIMESTAMP_COLUMN: "_future_timestamp",
            CURRENT_HUI_COLUMN: target_column,
        }
    )

    merged = current.merge(
        future,
        on=[
            HIVE_COLUMN,
            SPLIT_COLUMN,
            SEGMENT_COLUMN,
            "_future_timestamp",
        ],
        how="left",
        validate="many_to_one",
    )
    return merged.drop(columns=[SEGMENT_COLUMN, "_future_timestamp"])


def _prediction_frame(path: Path, expected_split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required prediction file not found: {path}")
    frame = pd.read_parquet(path)
    _require_columns(
        frame,
        {
            TIMESTAMP_COLUMN,
            HIVE_COLUMN,
            SPLIT_COLUMN,
            TARGET_COLUMN,
            RAW_PROBABILITY_COLUMN,
            CALIBRATED_PROBABILITY_COLUMN,
        },
        frame_name=f"{expected_split} calibrated predictions",
    )
    frame = frame.copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        frame[TIMESTAMP_COLUMN],
        errors="raise",
    )
    if not frame[SPLIT_COLUMN].astype(str).eq(expected_split).all():
        raise ValueError(
            f"Prediction file for {expected_split} contains other splits."
        )
    key_columns = [TIMESTAMP_COLUMN, HIVE_COLUMN, SPLIT_COLUMN]
    if frame.duplicated(key_columns).any():
        raise ValueError(
            f"Prediction file for {expected_split} has duplicate keys."
        )
    return frame[
        key_columns
        + [
            TARGET_COLUMN,
            RAW_PROBABILITY_COLUMN,
            CALIBRATED_PROBABILITY_COLUMN,
        ]
    ]


def build_classifier_derived_hui_dataset_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = config["classifier_derived_hui"]

    feature_path = _resolve_path(root, settings["feature_dataset_path"])
    output_path = _resolve_path(root, settings["hui_dataset_path"])
    output_directory = _resolve_path(root, settings["output_directory"])

    features = pd.read_parquet(feature_path)
    _require_columns(
        features,
        {
            TIMESTAMP_COLUMN,
            HIVE_COLUMN,
            SPLIT_COLUMN,
            TARGET_COLUMN,
        },
        frame_name="Reviewed feature dataset",
    )
    features = features.copy()
    features[TIMESTAMP_COLUMN] = pd.to_datetime(
        features[TIMESTAMP_COLUMN],
        errors="raise",
    )

    prediction_config = settings["calibrated_prediction_paths"]
    prediction_frames = [
        _prediction_frame(
            _resolve_path(root, prediction_config[split]),
            split,
        )
        for split in ("train", "validation", "test")
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)

    key_columns = [TIMESTAMP_COLUMN, HIVE_COLUMN, SPLIT_COLUMN]
    merged = features.merge(
        predictions,
        on=key_columns,
        how="left",
        suffixes=("", "_prediction"),
        validate="one_to_one",
    )

    if merged[CALIBRATED_PROBABILITY_COLUMN].isna().any():
        missing_count = int(
            merged[CALIBRATED_PROBABILITY_COLUMN].isna().sum()
        )
        raise ValueError(
            f"{missing_count} feature rows lack calibrated predictions."
        )

    prediction_target = f"{TARGET_COLUMN}_prediction"
    if prediction_target in merged.columns:
        mismatched = merged[TARGET_COLUMN].ne(merged[prediction_target])
        if mismatched.any():
            raise ValueError(
                "Prediction targets do not match the feature dataset."
            )
        merged = merged.drop(columns=[prediction_target])

    anchor_config = settings["hui_anchors"]
    probability_anchors = [
        float(item["calibrated_score"]) for item in anchor_config
    ]
    hui_anchors = [float(item["hui"]) for item in anchor_config]
    merged[CURRENT_HUI_COLUMN] = probability_to_hui(
        merged[CALIBRATED_PROBABILITY_COLUMN],
        probability_anchors=probability_anchors,
        hui_anchors=hui_anchors,
    )

    class_config = settings["classes"]
    merged[CURRENT_CLASS_COLUMN] = assign_harvest_readiness_class(
        merged[CURRENT_HUI_COLUMN],
        not_ready_upper=float(class_config["not_ready_upper"]),
        approaching_upper=float(class_config["approaching_upper"]),
        ready_upper=float(class_config["ready_upper"]),
    )

    merged = add_hui_history_features(merged)
    for horizon in settings["horizons_hours"]:
        merged = add_future_hui_target(
            merged,
            horizon_hours=int(horizon),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    definition = {
        "name": "Classifier-derived Provisional Harvest Urgency Index",
        "range": [0.0, 100.0],
        "source": (
            "Platt-adjusted 72-hour classifier score transformed by "
            "a monotonic piecewise-linear mapping fitted only from "
            "training out-of-fold evidence."
        ),
        "probability_interpretation": (
            "HUI is a relative urgency index, not a literal probability "
            "percentage and not a direct biological honey-maturity label."
        ),
        "calibration_status": "research_gate_not_passed",
        "anchors": anchor_config,
        "classes": class_config,
        "horizons_hours": [
            int(value) for value in settings["horizons_hours"]
        ],
        "history_features": HUI_HISTORY_COLUMNS,
    }
    _write_json(
        output_directory / "classifier_derived_hui_definition.json",
        definition,
    )

    distribution = (
        merged.groupby(
            [SPLIT_COLUMN, CURRENT_CLASS_COLUMN],
            observed=True,
        )
        .agg(
            rows=(CURRENT_HUI_COLUMN, "size"),
            mean_hui=(CURRENT_HUI_COLUMN, "mean"),
            median_hui=(CURRENT_HUI_COLUMN, "median"),
            minimum_hui=(CURRENT_HUI_COLUMN, "min"),
            maximum_hui=(CURRENT_HUI_COLUMN, "max"),
        )
        .reset_index()
    )
    distribution.to_csv(
        output_directory / "classifier_derived_hui_distribution.csv",
        index=False,
    )

    target_counts = []
    for split, group in merged.groupby(SPLIT_COLUMN, observed=True):
        record: dict[str, Any] = {"split": str(split), "rows": len(group)}
        for horizon in settings["horizons_hours"]:
            target_column = (
                f"future_classifier_derived_hui_{int(horizon)}h"
            )
            record[f"available_target_rows_{int(horizon)}h"] = int(
                group[target_column].notna().sum()
            )
        target_counts.append(record)
    pd.DataFrame(target_counts).to_csv(
        output_directory / "future_hui_target_availability.csv",
        index=False,
    )

    return {
        "status": "classifier_derived_hui_dataset_built",
        "rows": len(merged),
        "hives": int(merged[HIVE_COLUMN].nunique()),
        "dataset_path": str(output_path),
        "definition_path": str(
            output_directory / "classifier_derived_hui_definition.json"
        ),
        "distribution_path": str(
            output_directory / "classifier_derived_hui_distribution.csv"
        ),
    }


def _load_feature_manifest(path: Path) -> list[str]:
    manifest = pd.read_csv(path)
    for candidate in ("feature_name", "feature", "column", "name"):
        if candidate in manifest.columns:
            return (
                manifest[candidate]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
    raise ValueError(
        "Feature manifest must contain feature_name, feature, column or name."
    )


def _build_feature_sets(
    manifest_features: list[str],
    *,
    settings: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    available = list(
        dict.fromkeys(
            [CURRENT_HUI_COLUMN, *manifest_features, *HUI_HISTORY_COLUMNS]
        )
    )
    output: dict[str, list[str]] = {}

    for name, config in settings.items():
        selected: list[str] = []
        if bool(config.get("include_hui_history", False)):
            selected.extend([CURRENT_HUI_COLUMN, *HUI_HISTORY_COLUMNS])
        if bool(config.get("include_all_manifest", False)):
            selected.extend(manifest_features)
        else:
            prefixes = [
                str(value) for value in config.get("include_prefixes", [])
            ]
            selected.extend(
                feature
                for feature in manifest_features
                if any(feature.startswith(prefix) for prefix in prefixes)
            )

        excluded = [
            str(value) for value in config.get("exclude_prefixes", [])
        ]
        selected = [
            feature
            for feature in list(dict.fromkeys(selected))
            if feature in available
            and feature not in NON_FEATURE_COLUMNS
            and not feature.startswith("future_")
            and not any(feature.startswith(prefix) for prefix in excluded)
        ]
        if not selected:
            raise ValueError(f"Feature set {name!r} is empty.")
        output[name] = selected
    return output


def _build_models(
    model_config: dict[str, dict[str, Any]],
    *,
    random_state: int,
) -> dict[str, Any]:
    models: dict[str, Any] = {}

    if bool(model_config["persistence"]["enabled"]):
        models["persistence"] = PersistenceHuiRegressor()

    if bool(model_config["ridge"]["enabled"]):
        models["ridge"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    Ridge(alpha=float(model_config["ridge"]["alpha"])),
                ),
            ]
        )

    if bool(model_config["random_forest"]["enabled"]):
        settings = model_config["random_forest"]
        models["random_forest"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=int(settings["n_estimators"]),
                        max_depth=int(settings["max_depth"]),
                        min_samples_leaf=int(settings["min_samples_leaf"]),
                        max_features=settings["max_features"],
                        max_samples=float(settings["max_samples"]),
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if bool(model_config["xgboost"]["enabled"]):
        from xgboost import XGBRegressor

        settings = model_config["xgboost"]
        models["xgboost"] = XGBRegressor(
            n_estimators=int(settings["n_estimators"]),
            learning_rate=float(settings["learning_rate"]),
            max_depth=int(settings["max_depth"]),
            min_child_weight=float(settings["min_child_weight"]),
            subsample=float(settings["subsample"]),
            colsample_bytree=float(settings["colsample_bytree"]),
            reg_alpha=float(settings["reg_alpha"]),
            reg_lambda=float(settings["reg_lambda"]),
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

    if bool(model_config["lightgbm"]["enabled"]):
        from lightgbm import LGBMRegressor

        settings = model_config["lightgbm"]
        models["lightgbm"] = LGBMRegressor(
            n_estimators=int(settings["n_estimators"]),
            learning_rate=float(settings["learning_rate"]),
            num_leaves=int(settings["num_leaves"]),
            max_depth=int(settings["max_depth"]),
            min_child_samples=int(settings["min_child_samples"]),
            subsample=float(settings["subsample"]),
            colsample_bytree=float(settings["colsample_bytree"]),
            reg_alpha=float(settings["reg_alpha"]),
            reg_lambda=float(settings["reg_lambda"]),
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    return models


def _class_codes(values: np.ndarray, class_config: dict[str, float]) -> np.ndarray:
    return np.digitize(
        np.asarray(values, dtype=float),
        bins=[
            float(class_config["not_ready_upper"]),
            float(class_config["approaching_upper"]),
            float(class_config["ready_upper"]),
        ],
        right=False,
    )


def regression_metrics(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
    *,
    class_config: dict[str, float],
) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(predicted, dtype=float).clip(0.0, 100.0)
    errors = prediction_array - actual_array

    return {
        "mae": float(mean_absolute_error(actual_array, prediction_array)),
        "rmse": float(
            mean_squared_error(actual_array, prediction_array) ** 0.5
        ),
        "median_absolute_error": float(
            median_absolute_error(actual_array, prediction_array)
        ),
        "bias": float(np.mean(errors)),
        "r2": float(r2_score(actual_array, prediction_array)),
        "within_5_points_fraction": float(
            np.mean(np.abs(errors) <= 5.0)
        ),
        "within_10_points_fraction": float(
            np.mean(np.abs(errors) <= 10.0)
        ),
        "readiness_class_agreement_fraction": float(
            np.mean(
                _class_codes(actual_array, class_config)
                == _class_codes(prediction_array, class_config)
            )
        ),
    }


def _downsample_training(
    frame: pd.DataFrame,
    *,
    maximum_rows: int,
) -> pd.DataFrame:
    if len(frame) <= maximum_rows:
        return frame.copy()
    positions = np.linspace(
        0,
        len(frame) - 1,
        maximum_rows,
        dtype=int,
    )
    return frame.iloc[positions].copy()


def _candidate_selection_key(
    candidate: CandidateEvaluation,
) -> tuple[float, float, float, int, int]:
    metrics = candidate.validation_metrics
    return (
        -float(metrics["mae"]),
        float(metrics["readiness_class_agreement_fraction"]),
        -float(metrics["median_absolute_error"]),
        -len(candidate.feature_columns),
        -MODEL_COMPLEXITY[candidate.model_name],
    )


def _fit_candidate(
    estimator: Any,
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    class_config: dict[str, float],
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    model_features = feature_columns
    if isinstance(estimator, PersistenceHuiRegressor):
        model_features = [CURRENT_HUI_COLUMN]

    fitted = copy.deepcopy(estimator)
    fitted.fit(train[model_features], train[target_column])
    validation_prediction = np.asarray(
        fitted.predict(validation[model_features]),
        dtype=float,
    ).clip(0.0, 100.0)
    test_prediction = np.asarray(
        fitted.predict(test[model_features]),
        dtype=float,
    ).clip(0.0, 100.0)

    validation_metrics = regression_metrics(
        validation[target_column],
        validation_prediction,
        class_config=class_config,
    )
    test_metrics = regression_metrics(
        test[target_column],
        test_prediction,
        class_config=class_config,
    )
    return (
        fitted,
        validation_prediction,
        test_prediction,
        validation_metrics,
        test_metrics,
    )


def evaluate_future_hui_research_gate(
    comparison: pd.DataFrame,
    summary: dict[str, Any],
    *,
    horizons_hours: list[int],
    minimum_improvement: float,
    required_improved_horizons: int,
    maximum_test_to_validation_ratio: float,
) -> dict[str, Any]:
    horizon_results: dict[str, Any] = {}
    passed_count = 0

    for horizon in horizons_hours:
        rows = comparison.loc[
            comparison["horizon_hours"].eq(horizon)
            & comparison["status"].eq("ok")
        ]
        persistence = rows.loc[rows["model"].eq("persistence")]
        if persistence.empty:
            raise ValueError(
                f"Persistence baseline is missing for {horizon}h."
            )
        baseline_mae = float(persistence["validation_mae"].min())
        selected = summary["horizons"][str(horizon)]
        selected_mae = float(selected["validation"]["mae"])
        test_mae = float(selected["test"]["mae"])

        improvement = (
            (baseline_mae - selected_mae) / baseline_mae
            if baseline_mae > 0.0
            else 0.0
        )
        ratio = (
            test_mae / selected_mae
            if selected_mae > 0.0
            else float("inf")
        )
        horizon_passed = bool(
            improvement >= minimum_improvement
            and ratio <= maximum_test_to_validation_ratio
        )
        passed_count += int(horizon_passed)

        horizon_results[str(horizon)] = {
            "selected_model": selected["selected_model"],
            "selected_feature_set": selected["selected_feature_set"],
            "persistence_validation_mae": baseline_mae,
            "selected_validation_mae": selected_mae,
            "selected_test_mae": test_mae,
            "validation_mae_improvement_fraction": improvement,
            "test_to_validation_mae_ratio": ratio,
            "horizon_passed": horizon_passed,
        }

    gate_passed = passed_count >= required_improved_horizons
    return {
        "status": (
            "classifier_derived_future_hui_gate_passed"
            if gate_passed
            else "classifier_derived_future_hui_gate_failed"
        ),
        "gate_passed": gate_passed,
        "improved_horizon_count": passed_count,
        "required_improved_horizons": required_improved_horizons,
        "minimum_validation_mae_improvement_fraction": minimum_improvement,
        "maximum_test_to_validation_mae_ratio": (
            maximum_test_to_validation_ratio
        ),
        "ready_for_viva_research_dashboard": True,
        "ready_for_operational_deployment": False,
        "horizons": horizon_results,
        "warning": (
            "The target is a classifier-derived research index. Gate passage "
            "would support a viva prototype, not independent biological or "
            "operational validation."
        ),
    }


def run_future_hui_regression_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = config["classifier_derived_hui"]
    regression = settings["future_regression"]

    dataset_path = _resolve_path(root, settings["hui_dataset_path"])
    manifest_path = _resolve_path(root, settings["feature_manifest_path"])
    output_directory = _resolve_path(root, settings["output_directory"])
    model_directory = _resolve_path(root, regression["model_directory"])

    data = pd.read_parquet(dataset_path)
    data[TIMESTAMP_COLUMN] = pd.to_datetime(
        data[TIMESTAMP_COLUMN],
        errors="raise",
    )
    manifest_features = _load_feature_manifest(manifest_path)
    feature_sets = _build_feature_sets(
        manifest_features,
        settings=regression["feature_sets"],
    )
    models = _build_models(
        regression["models"],
        random_state=int(regression["random_state"]),
    )
    class_config = {
        str(key): float(value)
        for key, value in settings["classes"].items()
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    comparison_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "research_stage": (
            "Classifier-derived current HUI and direct multi-horizon future "
            "HUI regression completed for viva evaluation."
        ),
        "target_status": "classifier_derived_provisional_research_index",
        "operational_use_allowed": False,
        "horizons": {},
    }

    for horizon_value in settings["horizons_hours"]:
        horizon = int(horizon_value)
        target_column = f"future_classifier_derived_hui_{horizon}h"
        _require_columns(
            data,
            {
                SPLIT_COLUMN,
                CURRENT_HUI_COLUMN,
                target_column,
            },
            frame_name="Classifier-derived HUI dataset",
        )

        usable = data.loc[data[target_column].notna()].copy()
        train = usable.loc[usable[SPLIT_COLUMN].eq("train")].copy()
        validation = usable.loc[
            usable[SPLIT_COLUMN].eq("validation")
        ].copy()
        test = usable.loc[usable[SPLIT_COLUMN].eq("test")].copy()

        if train.empty or validation.empty or test.empty:
            raise ValueError(
                f"Train, validation and test targets are required for {horizon}h."
            )

        train = _downsample_training(
            train,
            maximum_rows=int(regression["maximum_training_rows"]),
        )

        evaluations: list[CandidateEvaluation] = []
        for model_name, estimator in models.items():
            applicable_feature_sets = (
                {"persistence": [CURRENT_HUI_COLUMN]}
                if model_name == "persistence"
                else feature_sets
            )

            for feature_set_name, feature_columns in applicable_feature_sets.items():
                missing_features = sorted(
                    set(feature_columns).difference(usable.columns)
                )
                if missing_features:
                    comparison_records.append(
                        {
                            "horizon_hours": horizon,
                            "model": model_name,
                            "feature_set": feature_set_name,
                            "status": "failed",
                            "error": f"Missing features: {missing_features}",
                        }
                    )
                    continue

                try:
                    (
                        fitted,
                        validation_prediction,
                        test_prediction,
                        validation_metrics,
                        test_metrics,
                    ) = _fit_candidate(
                        estimator,
                        train=train,
                        validation=validation,
                        test=test,
                        feature_columns=feature_columns,
                        target_column=target_column,
                        class_config=class_config,
                    )
                    evaluation = CandidateEvaluation(
                        horizon_hours=horizon,
                        model_name=model_name,
                        feature_set_name=feature_set_name,
                        feature_columns=list(feature_columns),
                        fitted_estimator=fitted,
                        validation_prediction=validation_prediction,
                        test_prediction=test_prediction,
                        validation_metrics=validation_metrics,
                        test_metrics=test_metrics,
                    )
                    evaluations.append(evaluation)

                    record: dict[str, Any] = {
                        "horizon_hours": horizon,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "status": "ok",
                        "feature_count": len(feature_columns),
                        "training_rows": len(train),
                        "validation_rows": len(validation),
                        "test_rows": len(test),
                    }
                    record.update(
                        {
                            f"validation_{key}": value
                            for key, value in validation_metrics.items()
                        }
                    )
                    record.update(
                        {
                            f"test_{key}": value
                            for key, value in test_metrics.items()
                        }
                    )
                    comparison_records.append(record)
                except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                    comparison_records.append(
                        {
                            "horizon_hours": horizon,
                            "model": model_name,
                            "feature_set": feature_set_name,
                            "status": "failed",
                            "error": str(error),
                        }
                    )

        if not evaluations:
            raise RuntimeError(
                f"No future-HUI regression candidate completed for {horizon}h."
            )

        selected = max(evaluations, key=_candidate_selection_key)
        model_path = (
            model_directory
            / f"selected_classifier_derived_hui_regressor_{horizon}h.joblib"
        )
        metadata_path = model_path.with_suffix(".json")
        joblib.dump(selected.fitted_estimator, model_path)
        _write_json(
            metadata_path,
            {
                "horizon_hours": horizon,
                "selected_model": selected.model_name,
                "selected_feature_set": selected.feature_set_name,
                "feature_columns": selected.feature_columns,
                "target_column": target_column,
                "score_status": "classifier_derived_provisional_research_index",
                "operational_use_allowed": False,
            },
        )

        def make_prediction_frame(
            frame: pd.DataFrame,
            prediction: np.ndarray,
            *,
            target_column_name: str,
        ) -> pd.DataFrame:
            output = frame[
                [
                    TIMESTAMP_COLUMN,
                    HIVE_COLUMN,
                    SPLIT_COLUMN,
                    CURRENT_HUI_COLUMN,
                    CURRENT_CLASS_COLUMN,
                    target_column_name,
                ]
            ].copy()
            output["predicted_future_hui"] = prediction
            output["prediction_error_points"] = (
                output["predicted_future_hui"]
                - output[target_column_name]
            )
            output["actual_future_class"] = assign_harvest_readiness_class(
                output[target_column_name],
                not_ready_upper=class_config["not_ready_upper"],
                approaching_upper=class_config["approaching_upper"],
                ready_upper=class_config["ready_upper"],
            )
            output["predicted_future_class"] = assign_harvest_readiness_class(
                output["predicted_future_hui"],
                not_ready_upper=class_config["not_ready_upper"],
                approaching_upper=class_config["approaching_upper"],
                ready_upper=class_config["ready_upper"],
            )
            return output

        validation_predictions = make_prediction_frame(
            validation,
            selected.validation_prediction,
            target_column_name=target_column,
        )
        test_predictions = make_prediction_frame(
            test,
            selected.test_prediction,
            target_column_name=target_column,
        )
        validation_predictions.to_parquet(
            output_directory
            / f"selected_validation_predictions_{horizon}h.parquet",
            index=False,
        )
        test_predictions.to_parquet(
            output_directory
            / f"selected_test_predictions_{horizon}h.parquet",
            index=False,
        )

        summary["horizons"][str(horizon)] = {
            "selected_model": selected.model_name,
            "selected_feature_set": selected.feature_set_name,
            "selected_feature_count": len(selected.feature_columns),
            "validation": selected.validation_metrics,
            "test": selected.test_metrics,
            "model_path": str(model_path),
        }

    comparison = pd.DataFrame(comparison_records)
    comparison_path = output_directory / "future_hui_regression_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    summary_path = output_directory / "future_hui_regression_summary.json"
    _write_json(summary_path, summary)

    gate_config = regression["research_gate"]
    gate = evaluate_future_hui_research_gate(
        comparison,
        summary,
        horizons_hours=[
            int(value) for value in settings["horizons_hours"]
        ],
        minimum_improvement=float(
            gate_config["minimum_validation_mae_improvement_fraction"]
        ),
        required_improved_horizons=int(
            gate_config["required_improved_horizons"]
        ),
        maximum_test_to_validation_ratio=float(
            gate_config["maximum_test_to_validation_mae_ratio"]
        ),
    )
    gate_path = output_directory / "future_hui_regression_gate.json"
    _write_json(gate_path, gate)

    return {
        "status": "classifier_derived_future_hui_regression_complete",
        "comparison_path": str(comparison_path),
        "summary_path": str(summary_path),
        "gate_path": str(gate_path),
        "gate_passed": bool(gate["gate_passed"]),
    }
