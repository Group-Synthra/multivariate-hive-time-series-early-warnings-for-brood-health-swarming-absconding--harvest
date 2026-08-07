from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from multivari.modules.harvesting.forecast_readiness import (
    evaluate_forecasting_research_gate,
)

HIVE_COLUMN = "hive_id"
TIMESTAMP_COLUMN = "timestamp"
SPLIT_COLUMN = "split"
WEIGHT_COLUMN = "weight_kg"


class PersistenceDeltaRegressor(RegressorMixin, BaseEstimator):
    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> PersistenceDeltaRegressor:
        del features, target
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features), dtype=float)


class RecentTrendDeltaRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        *,
        trend_feature: str,
        horizon_hours: int,
    ) -> None:
        self.trend_feature = trend_feature
        self.horizon_hours = horizon_hours

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> RecentTrendDeltaRegressor:
        del target
        if self.trend_feature not in features.columns:
            raise ValueError(f"Trend feature not found: {self.trend_feature}")
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features[self.trend_feature].to_numpy(dtype=float) * self.horizon_hours


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


def build_feature_sets(
    available_features: list[str],
    feature_set_config: dict[str, Any],
) -> dict[str, list[str]]:
    available = list(dict.fromkeys(available_features))
    result: dict[str, list[str]] = {}

    for name, settings in feature_set_config.items():
        if bool(settings.get("include_all", False)):
            selected = available.copy()
        else:
            prefixes = [str(value) for value in settings.get("include_prefixes", [])]
            selected = [
                feature
                for feature in available
                if any(feature.startswith(prefix) for prefix in prefixes)
            ]

        excluded_prefixes = [str(value) for value in settings.get("exclude_prefixes", [])]
        selected = [
            feature
            for feature in selected
            if not any(feature.startswith(prefix) for prefix in excluded_prefixes)
        ]
        selected = list(dict.fromkeys(selected))
        if not selected:
            raise ValueError(f"Feature set '{name}' contains no features.")
        result[name] = selected

    return result


def _add_contiguous_segment_id(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values([HIVE_COLUMN, SPLIT_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)
    elapsed = (
        result.groupby([HIVE_COLUMN, SPLIT_COLUMN])[TIMESTAMP_COLUMN]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )
    starts_segment = elapsed.isna() | elapsed.ne(1.0)
    result["_segment_id"] = starts_segment.cumsum().astype("int64")
    return result


def build_robust_future_targets(
    feature_rows: pd.DataFrame,
    clean_rows: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    horizons_hours: list[int],
    target_window_hours: int,
) -> pd.DataFrame:
    """
    Build robust future deltas using trailing median endpoint windows.

    At time t, the reference is the median weight over the contiguous window
    ending at t. For horizon h, the future endpoint is the median over the
    contiguous window ending at t+h. Both windows remain in the same hive and
    official split. This reduces single-hour sensor noise without changing the
    observable forecasting target.
    """
    if target_window_hours <= 0:
        raise ValueError("target_window_hours must be greater than zero")

    _require_columns(
        feature_rows,
        {HIVE_COLUMN, TIMESTAMP_COLUMN, SPLIT_COLUMN},
        frame_name="Feature dataset",
    )
    _require_columns(
        clean_rows,
        {HIVE_COLUMN, TIMESTAMP_COLUMN, WEIGHT_COLUMN},
        frame_name="Clean dataset",
    )
    _require_columns(
        split_manifest,
        {HIVE_COLUMN, TIMESTAMP_COLUMN, SPLIT_COLUMN},
        frame_name="Split manifest",
    )

    features = feature_rows.copy()
    clean = clean_rows.copy()
    manifest = split_manifest.copy()
    for frame in (features, clean, manifest):
        frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="raise")

    clean_with_split = clean[[HIVE_COLUMN, TIMESTAMP_COLUMN, WEIGHT_COLUMN]].merge(
        manifest[[HIVE_COLUMN, TIMESTAMP_COLUMN, SPLIT_COLUMN]],
        on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        how="inner",
        validate="one_to_one",
    )
    clean_with_split = _add_contiguous_segment_id(clean_with_split)
    clean_with_split["robust_endpoint_weight_kg"] = clean_with_split.groupby(
        [HIVE_COLUMN, SPLIT_COLUMN, "_segment_id"],
        sort=False,
    )[WEIGHT_COLUMN].transform(
        lambda values: values.rolling(
            window=target_window_hours,
            min_periods=target_window_hours,
        ).median()
    )

    endpoint = clean_with_split[
        [
            HIVE_COLUMN,
            TIMESTAMP_COLUMN,
            SPLIT_COLUMN,
            "robust_endpoint_weight_kg",
        ]
    ].copy()
    result = features.merge(
        endpoint.rename(columns={"robust_endpoint_weight_kg": ("robust_reference_weight_kg")}),
        on=[HIVE_COLUMN, TIMESTAMP_COLUMN, SPLIT_COLUMN],
        how="left",
        validate="one_to_one",
    )

    for horizon in horizons_hours:
        if horizon <= 0:
            raise ValueError("Forecast horizons must be positive.")
        future = endpoint.copy()
        future[TIMESTAMP_COLUMN] = future[TIMESTAMP_COLUMN] - pd.Timedelta(hours=horizon)
        future_column = f"robust_future_weight_{horizon}h_kg"
        future = future.rename(columns={"robust_endpoint_weight_kg": future_column})
        result = result.merge(
            future,
            on=[HIVE_COLUMN, TIMESTAMP_COLUMN, SPLIT_COLUMN],
            how="left",
            validate="one_to_one",
        )
        result[f"robust_weight_delta_next_{horizon}h_kg"] = (
            result[future_column] - result["robust_reference_weight_kg"]
        )

    return result


def calculate_regression_metrics(
    target: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(target, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    error = predicted - actual
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
        "bias": float(error.mean()),
        "r2": float(r2_score(actual, predicted)),
        "within_0_5kg_fraction": float((np.abs(error) <= 0.5).mean()),
        "within_1kg_fraction": float((np.abs(error) <= 1.0).mean()),
    }


def _make_estimator(
    model_name: str,
    settings: dict[str, Any],
    *,
    horizon_hours: int,
    random_state: int,
) -> BaseEstimator:
    if model_name == "persistence":
        return PersistenceDeltaRegressor()
    if model_name == "recent_trend":
        return RecentTrendDeltaRegressor(
            trend_feature=str(settings["trend_feature"]),
            horizon_hours=horizon_hours,
        )
    if model_name == "ridge":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=float(settings["alpha"]))),
            ]
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(settings["n_estimators"]),
            max_depth=int(settings["max_depth"]),
            min_samples_leaf=int(settings["min_samples_leaf"]),
            max_features=settings["max_features"],
            max_samples=float(settings["max_samples"]),
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost") from error
        return XGBRegressor(
            n_estimators=int(settings["n_estimators"]),
            learning_rate=float(settings["learning_rate"]),
            max_depth=int(settings["max_depth"]),
            min_child_weight=float(settings["min_child_weight"]),
            subsample=float(settings["subsample"]),
            colsample_bytree=float(settings["colsample_bytree"]),
            reg_alpha=float(settings["reg_alpha"]),
            reg_lambda=float(settings["reg_lambda"]),
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm") from error
        return LGBMRegressor(
            n_estimators=int(settings["n_estimators"]),
            learning_rate=float(settings["learning_rate"]),
            num_leaves=int(settings["num_leaves"]),
            max_depth=int(settings["max_depth"]),
            min_child_samples=int(settings["min_child_samples"]),
            subsample=float(settings["subsample"]),
            colsample_bytree=float(settings["colsample_bytree"]),
            reg_alpha=float(settings["reg_alpha"]),
            reg_lambda=float(settings["reg_lambda"]),
            objective="regression",
            n_jobs=-1,
            random_state=random_state,
            verbosity=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _sample_training_rows(
    rows: pd.DataFrame,
    *,
    maximum_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if maximum_rows <= 0 or len(rows) <= maximum_rows:
        return rows.copy()
    return rows.sample(
        n=maximum_rows,
        random_state=random_state,
        replace=False,
    ).sort_index()


def _feature_manifest_names(feature_manifest: pd.DataFrame) -> list[str]:
    for column in ("feature_name", "feature"):
        if column in feature_manifest.columns:
            return feature_manifest[column].astype(str).tolist()
    raise ValueError("Feature manifest must contain 'feature_name' or 'feature'.")


def run_robust_weight_forecasting_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = config["robust_weight_forecasting"]

    feature_rows = pd.read_parquet(_resolve_path(root, settings["feature_dataset_path"]))
    clean_rows = pd.read_parquet(_resolve_path(root, settings["clean_data_path"]))
    split_manifest = pd.read_parquet(_resolve_path(root, settings["split_manifest_path"]))
    feature_manifest = pd.read_csv(_resolve_path(root, settings["feature_manifest_path"]))
    output_directory = _resolve_path(root, settings["output_directory"])
    model_directory = _resolve_path(root, settings["model_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    horizons = [int(value) for value in settings["horizons_hours"]]
    random_state = int(settings["random_state"])
    maximum_training_rows = int(settings["maximum_training_rows"])
    feature_sets = build_feature_sets(
        _feature_manifest_names(feature_manifest),
        settings["feature_sets"],
    )
    dataset = build_robust_future_targets(
        feature_rows,
        clean_rows,
        split_manifest,
        horizons_hours=horizons,
        target_window_hours=int(settings["target_window_hours"]),
    )

    comparison_records: list[dict[str, Any]] = []
    summary_by_horizon: dict[str, Any] = {}

    for horizon in horizons:
        target_column = f"robust_weight_delta_next_{horizon}h_kg"
        rows = dataset.loc[dataset[target_column].notna()].copy()
        train = rows.loc[rows[SPLIT_COLUMN].eq("train")]
        validation = rows.loc[rows[SPLIT_COLUMN].eq("validation")]
        test = rows.loc[rows[SPLIT_COLUMN].eq("test")]
        if train.empty or validation.empty or test.empty:
            raise ValueError(f"Horizon {horizon}h lacks train, validation or test rows.")

        candidates: list[
            tuple[
                tuple[float, float, float, int],
                str,
                str,
                list[str],
                BaseEstimator,
                dict[str, float],
            ]
        ] = []

        for feature_set_name, selected_features in feature_sets.items():
            complete_columns = selected_features + [target_column]
            model_train = train.dropna(subset=complete_columns)
            model_validation = validation.dropna(subset=complete_columns)
            if model_train.empty or model_validation.empty:
                continue

            for model_name, model_settings in settings["models"].items():
                if not bool(model_settings.get("enabled", True)):
                    continue
                try:
                    estimator = _make_estimator(
                        model_name,
                        model_settings,
                        horizon_hours=horizon,
                        random_state=random_state,
                    )
                    sampled_train = _sample_training_rows(
                        model_train,
                        maximum_rows=maximum_training_rows,
                        random_state=random_state,
                    )
                    started = time.perf_counter()
                    estimator.fit(
                        sampled_train[selected_features],
                        sampled_train[target_column],
                    )
                    training_seconds = time.perf_counter() - started
                    validation_prediction = estimator.predict(model_validation[selected_features])
                    validation_metrics = calculate_regression_metrics(
                        model_validation[target_column],
                        validation_prediction,
                    )
                    complexity = {
                        "persistence": 0,
                        "recent_trend": 1,
                        "ridge": 2,
                        "random_forest": 3,
                        "xgboost": 4,
                        "lightgbm": 5,
                    }[model_name]
                    key = (
                        validation_metrics["mae"],
                        validation_metrics["median_absolute_error"],
                        abs(validation_metrics["bias"]),
                        complexity,
                    )
                    candidates.append(
                        (
                            key,
                            model_name,
                            feature_set_name,
                            selected_features,
                            estimator,
                            validation_metrics,
                        )
                    )
                    comparison_records.append(
                        {
                            "horizon_hours": horizon,
                            "model": model_name,
                            "feature_set": feature_set_name,
                            "status": "ok",
                            "feature_count": len(selected_features),
                            "training_rows": len(sampled_train),
                            "validation_rows": len(model_validation),
                            "training_seconds": training_seconds,
                            **{
                                f"validation_{name}": value
                                for name, value in validation_metrics.items()
                            },
                        }
                    )
                except ImportError as error:
                    comparison_records.append(
                        {
                            "horizon_hours": horizon,
                            "model": model_name,
                            "feature_set": feature_set_name,
                            "status": "missing_dependency",
                            "error": str(error),
                        }
                    )
                except (ValueError, TypeError, RuntimeError, OSError) as error:
                    comparison_records.append(
                        {
                            "horizon_hours": horizon,
                            "model": model_name,
                            "feature_set": feature_set_name,
                            "status": "failed",
                            "error": str(error),
                        }
                    )

        if not candidates:
            raise RuntimeError(f"No robust forecasting candidate completed at {horizon}h.")

        (
            _,
            selected_model,
            selected_feature_set,
            selected_features,
            selected_estimator,
            validation_metrics,
        ) = min(candidates, key=lambda item: item[0])

        validation_complete = validation.dropna(subset=selected_features + [target_column])
        test_complete = test.dropna(subset=selected_features + [target_column])
        validation_prediction = selected_estimator.predict(validation_complete[selected_features])
        test_prediction = selected_estimator.predict(test_complete[selected_features])
        test_metrics = calculate_regression_metrics(test_complete[target_column], test_prediction)

        def build_predictions(
            frame: pd.DataFrame,
            prediction: np.ndarray,
            *,
            target_column_name: str,
        ) -> pd.DataFrame:
            result = frame[
                [
                    TIMESTAMP_COLUMN,
                    HIVE_COLUMN,
                    SPLIT_COLUMN,
                    "robust_reference_weight_kg",
                    target_column_name,
                ]
            ].copy()
            result = result.rename(
                columns={
                    target_column_name: "actual_delta_kg",
                }
            )
            result["predicted_delta_kg"] = prediction
            result["actual_future_weight_kg"] = (
                result["robust_reference_weight_kg"] + result["actual_delta_kg"]
            )
            result["predicted_future_weight_kg"] = (
                result["robust_reference_weight_kg"] + result["predicted_delta_kg"]
            )
            result["error_kg"] = result["predicted_delta_kg"] - result["actual_delta_kg"]
            return result

        build_predictions(
            validation_complete,
            validation_prediction,
            target_column_name=target_column,
        ).to_parquet(
            output_directory / f"selected_validation_predictions_{horizon}h.parquet",
            index=False,
        )
        build_predictions(
            test_complete,
            test_prediction,
            target_column_name=target_column,
        ).to_parquet(
            output_directory / f"selected_test_predictions_{horizon}h.parquet",
            index=False,
        )

        joblib.dump(
            selected_estimator,
            model_directory / f"selected_weight_forecaster_{horizon}h.joblib",
        )
        _write_json(
            model_directory / f"selected_weight_forecaster_{horizon}h.json",
            {
                "horizon_hours": horizon,
                "model": selected_model,
                "feature_set": selected_feature_set,
                "feature_columns": selected_features,
                "target": target_column,
                "target_definition": (
                    "Trailing-median endpoint weight at t+h minus "
                    "trailing-median endpoint weight at t."
                ),
            },
        )

        summary_by_horizon[str(horizon)] = {
            "selected_model": selected_model,
            "selected_feature_set": selected_feature_set,
            "selected_feature_count": len(selected_features),
            "training_rows": len(train),
            "validation_rows": len(validation_complete),
            "test_rows": len(test_complete),
            "validation": validation_metrics,
            "test": test_metrics,
        }

    comparison = pd.DataFrame(comparison_records)
    comparison.to_csv(
        output_directory / "robust_weight_forecasting_comparison.csv",
        index=False,
    )
    summary = {
        "research_stage": "Robust future hive-weight forecasting.",
        "target_window_hours": int(settings["target_window_hours"]),
        "target_definition": (
            "The target is the difference between a trailing-median "
            "weight endpoint at t+h and a trailing-median endpoint at t."
        ),
        "selection_rule": (
            "Minimize validation MAE, then median absolute error, "
            "absolute bias and model complexity."
        ),
        "horizons": summary_by_horizon,
        "warnings": [
            (
                "This reformulation reduces endpoint sensor noise but "
                "does not verify honey maturity."
            ),
            ("The readiness prototype remains blocked unless the research gate passes."),
        ],
    }
    _write_json(
        output_directory / "robust_weight_forecasting_summary.json",
        summary,
    )
    _write_json(
        output_directory / "robust_weight_target_audit.json",
        {
            "source_feature_rows": len(feature_rows),
            "target_window_hours": int(settings["target_window_hours"]),
            "horizons_hours": horizons,
            "target_rows_by_split_and_horizon": [
                {
                    "split": split,
                    "horizon_hours": horizon,
                    "rows": int(
                        dataset.loc[
                            dataset[SPLIT_COLUMN].eq(split)
                            & dataset[f"robust_weight_delta_next_{horizon}h_kg"].notna()
                        ].shape[0]
                    ),
                }
                for horizon in horizons
                for split in ("train", "validation", "test")
            ],
            "leakage_control": (
                "Both median windows are contiguous, hive-specific and "
                "contained inside the same official split."
            ),
        },
    )

    return {
        "status": "completed",
        "horizons": summary_by_horizon,
        "comparison_path": str(output_directory / "robust_weight_forecasting_comparison.csv"),
        "summary_path": str(output_directory / "robust_weight_forecasting_summary.json"),
    }


def run_robust_forecasting_research_gate_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = config["robust_weight_forecasting"]
    output_directory = _resolve_path(root, settings["output_directory"])
    comparison = pd.read_csv(output_directory / "robust_weight_forecasting_comparison.csv")
    summary = _read_json(output_directory / "robust_weight_forecasting_summary.json")
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
    result["target_definition"] = summary["target_definition"]
    result["research_decision"] = (
        "Proceed to the provisional readiness prototype only when "
        "ready_for_readiness_prototype is true. Do not lower the gate "
        "after seeing these results."
    )
    _write_json(
        _resolve_path(root, settings["research_gate_path"]),
        result,
    )
    return result
