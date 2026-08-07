from __future__ import annotations

import copy
import json
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
CURRENT_HUI_COLUMN = "provisional_hui"
CURRENT_CLASS_COLUMN = "provisional_hui_class"

NON_FEATURE_COLUMNS = {
    TIMESTAMP_COLUMN,
    HIVE_COLUMN,
    SPLIT_COLUMN,
    CURRENT_HUI_COLUMN,
    CURRENT_CLASS_COLUMN,
    "harvest_within_next_72h_reviewed",
    "harvest_within_next_72h",
    "target",
}


class PersistenceHuiRegressor:
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

    def predict(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        return (
            features[CURRENT_HUI_COLUMN]
            .to_numpy(dtype=float)
            .clip(0.0, 100.0)
        )


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
        json.dumps(
            payload,
            indent=2,
            default=_json_safe,
        ),
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


def assign_provisional_hui_class(
    values: pd.Series,
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
            "HUI class boundaries must be increasing inside 0–100."
        )

    labels = np.select(
        [
            values.lt(not_ready_upper),
            values.lt(approaching_upper),
            values.lt(ready_upper),
        ],
        [
            "Not Ready",
            "Approaching Harvest",
            "Ready — Inspection Recommended",
        ],
        default="High-Priority Harvest Review",
    )
    return pd.Series(labels, index=values.index, dtype="string")


def _robust_bounds(
    values: pd.Series,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float]:
    finite = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if finite.empty:
        raise ValueError(
            "Cannot estimate normalization bounds from empty values."
        )

    lower = float(finite.quantile(lower_quantile))
    upper = float(finite.quantile(upper_quantile))
    if upper <= lower:
        upper = lower + max(abs(lower) * 0.01, 1e-6)
    return lower, upper


def _scale_higher(
    values: pd.Series,
    *,
    lower: float,
    upper: float,
) -> pd.Series:
    return ((values - lower) / (upper - lower)).clip(0.0, 1.0)


def _scale_lower(
    values: pd.Series,
    *,
    lower: float,
    upper: float,
) -> pd.Series:
    return 1.0 - _scale_higher(
        values,
        lower=lower,
        upper=upper,
    )


def build_current_provisional_hui(
    frame: pd.DataFrame,
    *,
    component_config: dict[str, dict[str, Any]],
    lower_quantile: float,
    upper_quantile: float,
    quality_columns: list[str],
    quality_penalty_per_flag: float,
    minimum_quality_factor: float,
    class_config: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        TIMESTAMP_COLUMN,
        HIVE_COLUMN,
        SPLIT_COLUMN,
    }
    required.update(
        str(settings["column"])
        for settings in component_config.values()
    )
    _require_columns(
        frame,
        required,
        frame_name="Feature dataset",
    )

    result = frame.copy()
    result[TIMESTAMP_COLUMN] = pd.to_datetime(
        result[TIMESTAMP_COLUMN],
        errors="raise",
    )

    train_mask = result[SPLIT_COLUMN].eq("train")
    if not train_mask.any():
        raise ValueError(
            "Training rows are required for HUI normalization."
        )

    total_weight = float(
        sum(
            float(settings["weight"])
            for settings in component_config.values()
        )
    )
    if not np.isclose(total_weight, 1.0):
        raise ValueError(
            "Provisional HUI component weights must sum to one."
        )

    weighted_score = pd.Series(
        0.0,
        index=result.index,
        dtype="float64",
    )
    parameters: dict[str, Any] = {
        "score_status": "provisional_research_index",
        "normalization_source_split": "train",
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "components": {},
    }

    for name, settings in component_config.items():
        column = str(settings["column"])
        direction = str(settings["direction"])
        weight = float(settings["weight"])

        raw = pd.to_numeric(
            result[column],
            errors="coerce",
        )
        if "clip_lower" in settings:
            raw = raw.clip(
                lower=float(settings["clip_lower"])
            )
        if direction == "lower_absolute":
            raw = raw.abs()
            effective_direction = "lower"
        else:
            effective_direction = direction

        lower, upper = _robust_bounds(
            raw.loc[train_mask],
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

        if effective_direction == "higher":
            component_score = _scale_higher(
                raw,
                lower=lower,
                upper=upper,
            )
        elif effective_direction == "lower":
            component_score = _scale_lower(
                raw,
                lower=lower,
                upper=upper,
            )
        else:
            raise ValueError(
                f"Unsupported HUI component direction: {direction}"
            )

        result[f"hui_component_{name}"] = (
            component_score * 100.0
        )
        weighted_score += component_score * weight

        parameters["components"][name] = {
            "column": column,
            "direction": direction,
            "weight": weight,
            "lower": lower,
            "upper": upper,
        }

    available_quality_columns = [
        column
        for column in quality_columns
        if column in result.columns
    ]
    missing_quality_columns = sorted(
        set(quality_columns).difference(
            available_quality_columns
        )
    )

    if available_quality_columns:
        quality_flags = (
            result[available_quality_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
            .sum(axis=1)
        )
    else:
        quality_flags = pd.Series(
            0.0,
            index=result.index,
            dtype="float64",
        )

    quality_factor = (
        1.0
        - quality_penalty_per_flag * quality_flags
    ).clip(lower=minimum_quality_factor, upper=1.0)

    result["hui_data_quality_factor"] = quality_factor
    result[CURRENT_HUI_COLUMN] = (
        weighted_score * quality_factor * 100.0
    ).clip(0.0, 100.0)
    result[CURRENT_CLASS_COLUMN] = assign_provisional_hui_class(
        result[CURRENT_HUI_COLUMN],
        not_ready_upper=float(class_config["not_ready_upper"]),
        approaching_upper=float(
            class_config["approaching_upper"]
        ),
        ready_upper=float(class_config["ready_upper"]),
    )

    parameters["quality"] = {
        "configured_columns": quality_columns,
        "available_columns": available_quality_columns,
        "missing_columns": missing_quality_columns,
        "quality_penalty_per_flag": quality_penalty_per_flag,
        "minimum_quality_factor": minimum_quality_factor,
    }
    parameters["classes"] = class_config
    parameters["warning"] = (
        "This score is an engineered provisional research index. "
        "It is not a calibrated harvest probability and does not "
        "directly measure honey maturity."
    )

    return result, parameters


def add_future_hui_target(
    frame: pd.DataFrame,
    *,
    horizon_hours: int,
) -> pd.DataFrame:
    _require_columns(
        frame,
        {
            TIMESTAMP_COLUMN,
            HIVE_COLUMN,
            SPLIT_COLUMN,
            CURRENT_HUI_COLUMN,
        },
        frame_name="Provisional HUI dataset",
    )

    current = frame.copy()
    current[TIMESTAMP_COLUMN] = pd.to_datetime(
        current[TIMESTAMP_COLUMN],
        errors="raise",
    )
    current = current.sort_values(
        [HIVE_COLUMN, TIMESTAMP_COLUMN]
    ).reset_index(drop=True)

    elapsed_hours = (
        current.groupby(HIVE_COLUMN)[TIMESTAMP_COLUMN]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )
    previous_split = (
        current.groupby(HIVE_COLUMN)[SPLIT_COLUMN]
        .shift()
    )
    starts_segment = (
        elapsed_hours.isna()
        | elapsed_hours.ne(1.0)
        | current[SPLIT_COLUMN].ne(previous_split)
    )
    current["_hui_segment_id"] = (
        starts_segment.cumsum().astype("int64")
    )
    current["_future_timestamp"] = (
        current[TIMESTAMP_COLUMN]
        + pd.to_timedelta(horizon_hours, unit="h")
    )

    future = current[
        [
            HIVE_COLUMN,
            SPLIT_COLUMN,
            "_hui_segment_id",
            TIMESTAMP_COLUMN,
            CURRENT_HUI_COLUMN,
        ]
    ].rename(
        columns={
            TIMESTAMP_COLUMN: "_future_timestamp",
            CURRENT_HUI_COLUMN: (
                f"future_provisional_hui_{horizon_hours}h"
            ),
        }
    )

    merged = current.merge(
        future,
        on=[
            HIVE_COLUMN,
            SPLIT_COLUMN,
            "_hui_segment_id",
            "_future_timestamp",
        ],
        how="left",
        validate="many_to_one",
    )
    return merged.drop(
        columns=[
            "_future_timestamp",
            "_hui_segment_id",
        ]
    )


def _load_feature_manifest(path: Path) -> list[str]:
    manifest = pd.read_csv(path)
    for candidate in (
        "feature_name",
        "feature",
        "column",
        "name",
    ):
        if candidate in manifest.columns:
            return (
                manifest[candidate]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
    raise ValueError(
        "Feature manifest must contain feature_name, feature, "
        "column or name."
    )


def _feature_sets(
    available_features: list[str],
    *,
    settings: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}

    for name, config in settings.items():
        if bool(config.get("include_all", False)):
            selected = list(available_features)
        else:
            prefixes = [
                str(value)
                for value in config.get(
                    "include_prefixes",
                    [],
                )
            ]
            selected = [
                feature
                for feature in available_features
                if any(
                    feature.startswith(prefix)
                    for prefix in prefixes
                )
            ]

        excluded = [
            str(value)
            for value in config.get(
                "exclude_prefixes",
                [],
            )
        ]
        if excluded:
            selected = [
                feature
                for feature in selected
                if not any(
                    feature.startswith(prefix)
                    for prefix in excluded
                )
            ]

        selected = [
            feature
            for feature in selected
            if feature not in NON_FEATURE_COLUMNS
            and not feature.startswith("future_")
            and not feature.startswith("hui_component_")
        ]
        if not selected:
            raise ValueError(
                f"Feature set {name!r} is empty."
            )
        output[name] = selected

    return output


def _build_models(
    model_config: dict[str, dict[str, Any]],
    *,
    random_state: int,
) -> dict[str, Any]:
    models: dict[str, Any] = {}

    if model_config["persistence"]["enabled"]:
        models["persistence"] = PersistenceHuiRegressor()

    if model_config["ridge"]["enabled"]:
        models["ridge"] = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                ("scaler", StandardScaler()),
                (
                    "model",
                    Ridge(
                        alpha=float(
                            model_config["ridge"]["alpha"]
                        )
                    ),
                ),
            ]
        )

    if model_config["random_forest"]["enabled"]:
        settings = model_config["random_forest"]
        models["random_forest"] = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=int(
                            settings["n_estimators"]
                        ),
                        max_depth=int(settings["max_depth"]),
                        min_samples_leaf=int(
                            settings["min_samples_leaf"]
                        ),
                        max_features=settings["max_features"],
                        max_samples=float(
                            settings["max_samples"]
                        ),
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if model_config["xgboost"]["enabled"]:
        from xgboost import XGBRegressor

        settings = model_config["xgboost"]
        models["xgboost"] = XGBRegressor(
            n_estimators=int(settings["n_estimators"]),
            learning_rate=float(settings["learning_rate"]),
            max_depth=int(settings["max_depth"]),
            min_child_weight=float(
                settings["min_child_weight"]
            ),
            subsample=float(settings["subsample"]),
            colsample_bytree=float(
                settings["colsample_bytree"]
            ),
            reg_alpha=float(settings["reg_alpha"]),
            reg_lambda=float(settings["reg_lambda"]),
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_config["lightgbm"]["enabled"]:
        from lightgbm import LGBMRegressor

        settings = model_config["lightgbm"]
        models["lightgbm"] = LGBMRegressor(
            n_estimators=int(settings["n_estimators"]),
            learning_rate=float(settings["learning_rate"]),
            num_leaves=int(settings["num_leaves"]),
            max_depth=int(settings["max_depth"]),
            min_child_samples=int(
                settings["min_child_samples"]
            ),
            subsample=float(settings["subsample"]),
            colsample_bytree=float(
                settings["colsample_bytree"]
            ),
            reg_alpha=float(settings["reg_alpha"]),
            reg_lambda=float(settings["reg_lambda"]),
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    return models


def _regression_metrics(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    errors = predicted_array - actual_array

    return {
        "mae": float(
            mean_absolute_error(
                actual_array,
                predicted_array,
            )
        ),
        "rmse": float(
            mean_squared_error(
                actual_array,
                predicted_array,
            )
            ** 0.5
        ),
        "median_absolute_error": float(
            median_absolute_error(
                actual_array,
                predicted_array,
            )
        ),
        "bias": float(np.mean(errors)),
        "r2": float(
            r2_score(actual_array, predicted_array)
        ),
        "within_5_points_fraction": float(
            np.mean(np.abs(errors) <= 5.0)
        ),
        "within_10_points_fraction": float(
            np.mean(np.abs(errors) <= 10.0)
        ),
    }


def _downsample_training(
    frame: pd.DataFrame,
    *,
    maximum_rows: int,
) -> pd.DataFrame:
    if len(frame) <= maximum_rows:
        return frame

    positions = np.linspace(
        0,
        len(frame) - 1,
        maximum_rows,
        dtype=int,
    )
    return frame.iloc[positions].copy()


def _fit_predict_candidate(
    estimator: Any,
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray]:
    model_features = list(feature_columns)
    if isinstance(estimator, PersistenceHuiRegressor):
        model_features = [CURRENT_HUI_COLUMN]

    estimator.fit(
        train[model_features],
        train[target_column],
    )
    validation_prediction = np.asarray(
        estimator.predict(validation[model_features]),
        dtype=float,
    ).clip(0.0, 100.0)
    test_prediction = np.asarray(
        estimator.predict(test[model_features]),
        dtype=float,
    ).clip(0.0, 100.0)

    metrics = {
        "validation": _regression_metrics(
            validation[target_column],
            validation_prediction,
        ),
        "test": _regression_metrics(
            test[target_column],
            test_prediction,
        ),
    }
    return (
        estimator,
        metrics,
        validation_prediction,
        test_prediction,
    )


def evaluate_hui_research_gate(
    comparison: pd.DataFrame,
    summary: dict[str, Any],
    *,
    horizons_hours: list[int],
    minimum_improvement: float,
    required_improved_horizons: int,
    maximum_test_to_validation_ratio: float,
) -> dict[str, Any]:
    horizon_results: dict[str, Any] = {}
    improved_count = 0

    for horizon in horizons_hours:
        rows = comparison.loc[
            comparison["horizon_hours"].eq(horizon)
            & comparison["status"].eq("ok")
        ]
        baseline_rows = rows.loc[
            rows["model"].eq("persistence")
        ]
        if baseline_rows.empty:
            raise ValueError(
                f"No persistence baseline exists for {horizon}h."
            )

        baseline_mae = float(
            baseline_rows["validation_mae"].min()
        )
        selected = summary["horizons"][str(horizon)]
        selected_mae = float(
            selected["validation"]["mae"]
        )
        test_mae = float(selected["test"]["mae"])

        improvement = (
            (baseline_mae - selected_mae) / baseline_mae
            if baseline_mae > 0
            else 0.0
        )
        ratio = (
            test_mae / selected_mae
            if selected_mae > 0
            else np.inf
        )
        passed = bool(
            improvement >= minimum_improvement
            and ratio <= maximum_test_to_validation_ratio
        )
        improved_count += int(passed)

        horizon_results[str(horizon)] = {
            "selected_model": selected["selected_model"],
            "selected_feature_set": (
                selected["selected_feature_set"]
            ),
            "persistence_validation_mae": baseline_mae,
            "selected_validation_mae": selected_mae,
            "selected_test_mae": test_mae,
            "validation_mae_improvement_fraction": (
                improvement
            ),
            "test_to_validation_mae_ratio": ratio,
            "horizon_passed": passed,
        }

    gate_passed = bool(
        improved_count >= required_improved_horizons
    )
    return {
        "status": (
            "provisional_hui_regression_gate_passed"
            if gate_passed
            else "provisional_hui_regression_gate_failed"
        ),
        "ready_for_research_dashboard": True,
        "ready_for_operational_hui": False,
        "gate_passed": gate_passed,
        "improved_horizon_count": improved_count,
        "required_improved_horizons": (
            required_improved_horizons
        ),
        "minimum_validation_mae_improvement_fraction": (
            minimum_improvement
        ),
        "maximum_test_to_validation_mae_ratio": (
            maximum_test_to_validation_ratio
        ),
        "horizons": horizon_results,
        "warning": (
            "Even when this regression gate passes, the target is "
            "an engineered Provisional HUI rather than verified "
            "honey maturity or a calibrated harvest probability."
        ),
    }


def build_provisional_hui_dataset_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    settings = config["provisional_hui_regression"]

    feature_path = _resolve_path(
        root,
        settings["feature_dataset_path"],
    )
    output_path = _resolve_path(
        root,
        settings["hui_dataset_path"],
    )
    output_directory = _resolve_path(
        root,
        settings["output_directory"],
    )

    features = pd.read_parquet(feature_path)
    normalization = settings["normalization"]
    hui, parameters = build_current_provisional_hui(
        features,
        component_config=settings["hui_components"],
        lower_quantile=float(
            normalization["lower_quantile"]
        ),
        upper_quantile=float(
            normalization["upper_quantile"]
        ),
        quality_columns=[
            str(value)
            for value in settings["quality_columns"]
        ],
        quality_penalty_per_flag=float(
            settings["quality_penalty_per_flag"]
        ),
        minimum_quality_factor=float(
            settings["minimum_quality_factor"]
        ),
        class_config={
            str(key): float(value)
            for key, value in settings["classes"].items()
        },
    )

    for horizon in settings["horizons_hours"]:
        hui = add_future_hui_target(
            hui,
            horizon_hours=int(horizon),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    hui.to_parquet(output_path, index=False)

    _write_json(
        output_directory
        / "provisional_hui_definition.json",
        parameters,
    )

    distribution = (
        hui.groupby(
            [SPLIT_COLUMN, CURRENT_CLASS_COLUMN],
            observed=True,
        )
        .agg(
            rows=(CURRENT_HUI_COLUMN, "size"),
            mean_hui=(CURRENT_HUI_COLUMN, "mean"),
            median_hui=(CURRENT_HUI_COLUMN, "median"),
        )
        .reset_index()
    )
    distribution.to_csv(
        output_directory
        / "provisional_hui_distribution.csv",
        index=False,
    )

    return {
        "status": "provisional_hui_dataset_built",
        "rows": len(hui),
        "hives": int(hui[HIVE_COLUMN].nunique()),
        "dataset_path": str(output_path),
        "definition_path": str(
            output_directory
            / "provisional_hui_definition.json"
        ),
    }


def run_provisional_hui_regression_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    settings = config["provisional_hui_regression"]

    dataset_path = _resolve_path(
        root,
        settings["hui_dataset_path"],
    )
    manifest_path = _resolve_path(
        root,
        settings["feature_manifest_path"],
    )
    output_directory = _resolve_path(
        root,
        settings["output_directory"],
    )
    model_directory = _resolve_path(
        root,
        settings["model_directory"],
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_parquet(dataset_path)
    dataset[TIMESTAMP_COLUMN] = pd.to_datetime(
        dataset[TIMESTAMP_COLUMN],
        errors="raise",
    )
    dataset = dataset.sort_values(
        [TIMESTAMP_COLUMN, HIVE_COLUMN]
    ).reset_index(drop=True)

    manifest_features = _load_feature_manifest(
        manifest_path
    )
    available_features = [
        feature
        for feature in manifest_features
        if feature in dataset.columns
    ]
    if not available_features:
        raise ValueError(
            "No feature-manifest columns exist in the HUI dataset."
        )

    feature_sets = _feature_sets(
        available_features,
        settings=settings["feature_sets"],
    )
    models = _build_models(
        settings["models"],
        random_state=int(settings["random_state"]),
    )

    comparison_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "research_stage": (
            "Direct multi-horizon regression of an engineered "
            "Provisional HUI target."
        ),
        "target_status": "provisional_research_index",
        "selection_rule": (
            "Minimize validation MAE, then median absolute error, "
            "absolute bias and model complexity."
        ),
        "horizons": {},
        "warning": (
            "The predicted target is not a calibrated probability "
            "and does not directly measure honey maturity."
        ),
    }

    complexity_order = {
        "persistence": 0,
        "ridge": 1,
        "random_forest": 2,
        "xgboost": 3,
        "lightgbm": 4,
    }

    for horizon_value in settings["horizons_hours"]:
        horizon = int(horizon_value)
        target_column = (
            f"future_provisional_hui_{horizon}h"
        )
        required = {
            TIMESTAMP_COLUMN,
            HIVE_COLUMN,
            SPLIT_COLUMN,
            CURRENT_HUI_COLUMN,
            target_column,
        }
        _require_columns(
            dataset,
            required,
            frame_name="Provisional HUI regression dataset",
        )

        modelling = dataset.loc[
            dataset[target_column].notna()
            & dataset[CURRENT_HUI_COLUMN].notna()
        ].copy()
        train = modelling.loc[
            modelling[SPLIT_COLUMN].eq("train")
        ]
        validation = modelling.loc[
            modelling[SPLIT_COLUMN].eq("validation")
        ]
        test = modelling.loc[
            modelling[SPLIT_COLUMN].eq("test")
        ]

        if train.empty or validation.empty or test.empty:
            raise ValueError(
                f"Missing train/validation/test rows for {horizon}h."
            )

        train_fit = _downsample_training(
            train,
            maximum_rows=int(
                settings["maximum_training_rows"]
            ),
        )

        candidates: list[dict[str, Any]] = []

        for model_name, estimator_template in models.items():
            candidate_feature_sets = (
                {"persistence": [CURRENT_HUI_COLUMN]}
                if model_name == "persistence"
                else feature_sets
            )

            for feature_set_name, feature_columns in (
                candidate_feature_sets.items()
            ):
                estimator = copy.deepcopy(
                    estimator_template
                )
                try:
                    (
                        fitted,
                        metrics,
                        validation_prediction,
                        test_prediction,
                    ) = _fit_predict_candidate(
                        estimator,
                        train=train_fit,
                        validation=validation,
                        test=test,
                        feature_columns=feature_columns,
                        target_column=target_column,
                    )
                    row = {
                        "horizon_hours": horizon,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "status": "ok",
                        "feature_count": len(feature_columns),
                        "training_rows": len(train_fit),
                        "validation_rows": len(validation),
                        "test_rows": len(test),
                        "validation_mae": metrics[
                            "validation"
                        ]["mae"],
                        "validation_rmse": metrics[
                            "validation"
                        ]["rmse"],
                        "validation_median_absolute_error": (
                            metrics["validation"][
                                "median_absolute_error"
                            ]
                        ),
                        "validation_bias": metrics[
                            "validation"
                        ]["bias"],
                        "validation_r2": metrics[
                            "validation"
                        ]["r2"],
                        "validation_within_5_points_fraction": (
                            metrics["validation"][
                                "within_5_points_fraction"
                            ]
                        ),
                        "validation_within_10_points_fraction": (
                            metrics["validation"][
                                "within_10_points_fraction"
                            ]
                        ),
                        "test_mae": metrics["test"]["mae"],
                        "test_rmse": metrics["test"]["rmse"],
                        "test_median_absolute_error": (
                            metrics["test"][
                                "median_absolute_error"
                            ]
                        ),
                        "test_bias": metrics["test"]["bias"],
                        "test_r2": metrics["test"]["r2"],
                        "test_within_5_points_fraction": (
                            metrics["test"][
                                "within_5_points_fraction"
                            ]
                        ),
                        "test_within_10_points_fraction": (
                            metrics["test"][
                                "within_10_points_fraction"
                            ]
                        ),
                        "_fitted": fitted,
                        "_validation_prediction": (
                            validation_prediction
                        ),
                        "_test_prediction": test_prediction,
                        "_feature_columns": feature_columns,
                        "_metrics": metrics,
                    }
                except (
                    ValueError,
                    TypeError,
                    RuntimeError,
                    OSError,
                ) as error:
                    row = {
                        "horizon_hours": horizon,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "status": "failed",
                        "error": str(error),
                    }

                comparison_rows.append(
                    {
                        key: value
                        for key, value in row.items()
                        if not key.startswith("_")
                    }
                )
                if row["status"] == "ok":
                    candidates.append(row)

        if not candidates:
            raise RuntimeError(
                f"All Provisional HUI models failed for {horizon}h."
            )

        selected = min(
            candidates,
            key=lambda row: (
                row["validation_mae"],
                row[
                    "validation_median_absolute_error"
                ],
                abs(row["validation_bias"]),
                complexity_order[row["model"]],
            ),
        )

        model_path = (
            model_directory
            / f"selected_provisional_hui_regressor_{horizon}h.joblib"
        )
        metadata_path = (
            model_directory
            / f"selected_provisional_hui_regressor_{horizon}h.json"
        )
        joblib.dump(selected["_fitted"], model_path)
        _write_json(
            metadata_path,
            {
                "horizon_hours": horizon,
                "selected_model": selected["model"],
                "selected_feature_set": (
                    selected["feature_set"]
                ),
                "feature_columns": (
                    selected["_feature_columns"]
                ),
                "target_column": target_column,
                "score_status": (
                    "provisional_research_index"
                ),
                "operational_use_allowed": False,
            },
        )

        def prediction_frame(
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
            output["predicted_future_provisional_hui"] = (
                prediction
            )
            output["prediction_error_points"] = (
                output[
                    "predicted_future_provisional_hui"
                ]
                - output[target_column_name]
            )
            return output

        validation_predictions = prediction_frame(
            validation,
            selected["_validation_prediction"],
            target_column_name=target_column,
        )
        test_predictions = prediction_frame(
            test,
            selected["_test_prediction"],
            target_column_name=target_column,
        )
        validation_predictions.to_parquet(
            output_directory
            / (
                "selected_validation_predictions_"
                f"{horizon}h.parquet"
            ),
            index=False,
        )
        test_predictions.to_parquet(
            output_directory
            / f"selected_test_predictions_{horizon}h.parquet",
            index=False,
        )

        summary["horizons"][str(horizon)] = {
            "selected_model": selected["model"],
            "selected_feature_set": (
                selected["feature_set"]
            ),
            "selected_feature_count": len(
                selected["_feature_columns"]
            ),
            "training_rows": len(train_fit),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "validation": selected["_metrics"][
                "validation"
            ],
            "test": selected["_metrics"]["test"],
        }

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = (
        output_directory
        / "provisional_hui_regression_comparison.csv"
    )
    summary_path = (
        output_directory
        / "provisional_hui_regression_summary.json"
    )
    comparison.to_csv(comparison_path, index=False)
    _write_json(summary_path, summary)

    gate_config = settings["research_gate"]
    gate = evaluate_hui_research_gate(
        comparison,
        summary,
        horizons_hours=[
            int(value)
            for value in settings["horizons_hours"]
        ],
        minimum_improvement=float(
            gate_config[
                "minimum_validation_mae_improvement_fraction"
            ]
        ),
        required_improved_horizons=int(
            gate_config["required_improved_horizons"]
        ),
        maximum_test_to_validation_ratio=float(
            gate_config[
                "maximum_test_to_validation_mae_ratio"
            ]
        ),
    )
    gate_path = (
        output_directory
        / "provisional_hui_regression_gate.json"
    )
    _write_json(gate_path, gate)

    return {
        "status": "provisional_hui_regression_complete",
        "comparison_path": str(comparison_path),
        "summary_path": str(summary_path),
        "gate_path": str(gate_path),
        "gate_passed": gate["gate_passed"],
    }


def export_provisional_hui_dashboard_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    settings = config["provisional_hui_regression"]
    output_directory = _resolve_path(
        root,
        settings["output_directory"],
    )
    frontend_path = _resolve_path(
        root,
        settings["frontend_output_path"],
    )
    dataset_path = _resolve_path(
        root,
        settings["hui_dataset_path"],
    )

    summary = _read_json(
        output_directory
        / "provisional_hui_regression_summary.json"
    )
    gate = _read_json(
        output_directory
        / "provisional_hui_regression_gate.json"
    )
    definition = _read_json(
        output_directory
        / "provisional_hui_definition.json"
    )

    horizons = [
        int(value)
        for value in settings["horizons_hours"]
    ]
    merged: pd.DataFrame | None = None

    for horizon in horizons:
        prediction_path = (
            output_directory
            / f"selected_test_predictions_{horizon}h.parquet"
        )
        predictions = pd.read_parquet(prediction_path)
        keep = predictions[
            [
                TIMESTAMP_COLUMN,
                HIVE_COLUMN,
                CURRENT_HUI_COLUMN,
                CURRENT_CLASS_COLUMN,
                "predicted_future_provisional_hui",
            ]
        ].rename(
            columns={
                "predicted_future_provisional_hui": (
                    f"predicted_hui_{horizon}h"
                )
            }
        )

        if merged is None:
            merged = keep
        else:
            merged = merged.merge(
                keep[
                    [
                        TIMESTAMP_COLUMN,
                        HIVE_COLUMN,
                        f"predicted_hui_{horizon}h",
                    ]
                ],
                on=[TIMESTAMP_COLUMN, HIVE_COLUMN],
                how="inner",
                validate="one_to_one",
            )

    if merged is None or merged.empty:
        raise ValueError(
            "No test predictions were available for dashboard export."
        )

    class_config = {
        str(key): float(value)
        for key, value in settings["classes"].items()
    }
    for horizon in horizons:
        merged[f"predicted_class_{horizon}h"] = (
            assign_provisional_hui_class(
                merged[f"predicted_hui_{horizon}h"],
                not_ready_upper=class_config[
                    "not_ready_upper"
                ],
                approaching_upper=class_config[
                    "approaching_upper"
                ],
                ready_upper=class_config["ready_upper"],
            )
        )

    merged[TIMESTAMP_COLUMN] = pd.to_datetime(
        merged[TIMESTAMP_COLUMN],
        errors="raise",
    )
    merged = merged.sort_values(
        [HIVE_COLUMN, TIMESTAMP_COLUMN]
    )
    rows_per_hive = int(
        settings["dashboard_rows_per_hive"]
    )
    recent = (
        merged.groupby(HIVE_COLUMN, group_keys=False)
        .tail(rows_per_hive)
        .reset_index(drop=True)
    )
    latest = (
        recent.groupby(HIVE_COLUMN, as_index=False)
        .tail(1)
        .sort_values(HIVE_COLUMN)
    )

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {
                key: _json_safe(value)
                for key, value in row.items()
            }
            for row in frame.to_dict(orient="records")
        ]

    payload = {
        "score_name": "Provisional Harvest Utilization Index",
        "score_abbreviation": "Provisional HUI",
        "score_status": "provisional_research_index",
        "operational_use_allowed": False,
        "class_thresholds": class_config,
        "horizons_hours": horizons,
        "models": summary["horizons"],
        "research_gate": gate,
        "definition": definition,
        "available_hives": sorted(
            recent[HIVE_COLUMN]
            .astype(str)
            .unique()
            .tolist()
        ),
        "latest_by_hive": records(latest),
        "historical_test_series": records(recent),
        "warnings": [
            (
                "The current and future values are predictions of "
                "an engineered Provisional HUI target."
            ),
            (
                "This is not a calibrated harvest probability and "
                "does not directly measure honey maturity."
            ),
            (
                "A beekeeper inspection is required before any "
                "harvesting decision."
            ),
        ],
        "source_dataset": str(dataset_path),
    }

    frontend_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    frontend_path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=_json_safe,
        ),
        encoding="utf-8",
    )

    return {
        "status": "provisional_hui_dashboard_exported",
        "output_path": str(frontend_path),
        "hive_count": len(payload["available_hives"]),
        "series_rows": len(recent),
    }
