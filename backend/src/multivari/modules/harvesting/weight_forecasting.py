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

HIVE_COLUMN = "hive_id"
TIMESTAMP_COLUMN = "timestamp"
SPLIT_COLUMN = "split"
WEIGHT_COLUMN = "weight_kg"
CURRENT_WEIGHT_FEATURE = "weight_kg_current"


class PersistenceDeltaRegressor(
    RegressorMixin,
    BaseEstimator,
):
    """Predict no change from the current observed hive weight."""

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> PersistenceDeltaRegressor:
        del features, target
        return self

    def predict(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        return np.zeros(len(features), dtype=float)


class RecentTrendDeltaRegressor(
    RegressorMixin,
    BaseEstimator,
):
    """Extend a past-only hourly weight trend over the forecast horizon."""

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
            raise ValueError(f"Recent-trend feature is unavailable: {self.trend_feature}")
        return self

    def predict(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        return features[self.trend_feature].to_numpy(dtype=float) * self.horizon_hours


def _resolve_path(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    frame_name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


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


def build_feature_sets(
    available_features: list[str],
    feature_set_config: dict[str, Any],
) -> dict[str, list[str]]:
    available = list(dict.fromkeys(available_features))
    available_set = set(available)
    result: dict[str, list[str]] = {}

    for name, settings in feature_set_config.items():
        if settings.get("include_all"):
            selected = available.copy()
        elif "include" in settings:
            requested = [str(value) for value in settings["include"]]
            missing = sorted(set(requested).difference(available_set))
            if missing:
                raise ValueError(f"Feature set '{name}' requests missing features: {missing}")
            selected = requested
        else:
            prefixes = [
                str(value)
                for value in settings.get(
                    "include_prefixes",
                    [],
                )
            ]
            selected = [
                feature
                for feature in available
                if any(feature.startswith(prefix) for prefix in prefixes)
            ]

        excluded_prefixes = [
            str(value)
            for value in settings.get(
                "exclude_prefixes",
                [],
            )
        ]
        if excluded_prefixes:
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


def build_future_weight_targets(
    feature_rows: pd.DataFrame,
    clean_rows: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    horizons_hours: list[int],
) -> pd.DataFrame:
    """
    Create exact future-weight targets without crossing hives or splits.

    The endpoint at t + horizon must exist exactly and must remain in the same
    official split as the feature row. The target is future weight minus the
    current observed weight.
    """
    _require_columns(
        feature_rows,
        {
            HIVE_COLUMN,
            TIMESTAMP_COLUMN,
            SPLIT_COLUMN,
            CURRENT_WEIGHT_FEATURE,
        },
        frame_name="Reviewed feature dataset",
    )
    _require_columns(
        clean_rows,
        {
            HIVE_COLUMN,
            TIMESTAMP_COLUMN,
            WEIGHT_COLUMN,
        },
        frame_name="Clean sensor dataset",
    )
    _require_columns(
        split_manifest,
        {
            HIVE_COLUMN,
            TIMESTAMP_COLUMN,
            SPLIT_COLUMN,
        },
        frame_name="Common split manifest",
    )

    features = feature_rows.copy()
    clean = clean_rows.copy()
    manifest = split_manifest.copy()

    for frame in (features, clean, manifest):
        frame[TIMESTAMP_COLUMN] = pd.to_datetime(
            frame[TIMESTAMP_COLUMN],
            errors="raise",
        )

    for frame, name in (
        (features, "feature rows"),
        (clean, "clean rows"),
        (manifest, "split manifest"),
    ):
        duplicates = int(frame.duplicated([HIVE_COLUMN, TIMESTAMP_COLUMN]).sum())
        if duplicates:
            raise ValueError(f"{name} contains duplicate hive-timestamp rows: {duplicates}")

    clean_with_split = clean[[HIVE_COLUMN, TIMESTAMP_COLUMN, WEIGHT_COLUMN]].merge(
        manifest[[HIVE_COLUMN, TIMESTAMP_COLUMN, SPLIT_COLUMN]],
        on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        how="inner",
        validate="one_to_one",
    )

    result = features.copy()

    for horizon in horizons_hours:
        if horizon <= 0:
            raise ValueError("Forecast horizons must be greater than zero.")

        future = clean_with_split.copy()
        future[TIMESTAMP_COLUMN] = future[TIMESTAMP_COLUMN] - pd.Timedelta(hours=horizon)
        future = future.rename(
            columns={
                WEIGHT_COLUMN: (f"future_weight_{horizon}h_kg"),
                SPLIT_COLUMN: f"future_split_{horizon}h",
            }
        )

        result = result.merge(
            future,
            on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
            how="left",
            validate="one_to_one",
        )

        future_split_column = f"future_split_{horizon}h"
        future_weight_column = f"future_weight_{horizon}h_kg"
        target_column = f"weight_delta_next_{horizon}h_kg"

        same_split = result[future_split_column].eq(result[SPLIT_COLUMN])
        result[target_column] = (
            result[future_weight_column] - result[CURRENT_WEIGHT_FEATURE]
        ).where(same_split)
        result = result.drop(columns=[future_split_column])

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
                (
                    "model",
                    Ridge(alpha=float(settings["alpha"])),
                ),
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

    raise ValueError(f"Unsupported forecasting model: {model_name}")


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


def _per_hive_metrics(
    predictions: pd.DataFrame,
    *,
    horizon_hours: int,
    split: str,
    model_name: str,
    feature_set_name: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for hive_id, group in predictions.groupby(
        HIVE_COLUMN,
        sort=True,
    ):
        metrics = calculate_regression_metrics(
            group["actual_delta_kg"],
            group["predicted_delta_kg"],
        )
        records.append(
            {
                HIVE_COLUMN: hive_id,
                SPLIT_COLUMN: split,
                "horizon_hours": horizon_hours,
                "model": model_name,
                "feature_set": feature_set_name,
                "rows": len(group),
                **metrics,
            }
        )
    return pd.DataFrame(records)


def run_weight_forecasting_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = config["weight_forecasting"]

    feature_path = _resolve_path(
        root,
        settings["feature_dataset_path"],
    )
    clean_path = _resolve_path(
        root,
        settings["clean_data_path"],
    )
    split_path = _resolve_path(
        root,
        settings["split_manifest_path"],
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

    feature_rows = pd.read_parquet(feature_path)
    clean_rows = pd.read_parquet(clean_path)
    split_manifest = pd.read_parquet(split_path)
    feature_manifest = pd.read_csv(manifest_path)

    available_features = feature_manifest["feature_name"].astype(str).tolist()
    feature_sets = build_feature_sets(
        available_features,
        settings["feature_sets"],
    )
    horizons = [int(value) for value in settings["horizons_hours"]]
    random_state = int(settings["random_state"])
    maximum_training_rows = int(settings["maximum_training_rows"])

    dataset = build_future_weight_targets(
        feature_rows,
        clean_rows,
        split_manifest,
        horizons_hours=horizons,
    )

    comparison_records: list[dict[str, Any]] = []
    summary_by_horizon: dict[str, Any] = {}
    all_per_hive: list[pd.DataFrame] = []

    for horizon in horizons:
        target_column = f"weight_delta_next_{horizon}h_kg"
        horizon_rows = dataset.loc[dataset[target_column].notna()].copy()

        train = horizon_rows.loc[horizon_rows[SPLIT_COLUMN].eq("train")]
        validation = horizon_rows.loc[horizon_rows[SPLIT_COLUMN].eq("validation")]
        test = horizon_rows.loc[horizon_rows[SPLIT_COLUMN].eq("test")]

        if train.empty or validation.empty or test.empty:
            raise ValueError(
                f"Horizon {horizon}h lacks train, validation or "
                "test rows after exact future-target construction."
            )

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
            for model_name, model_settings in settings["models"].items():
                if not bool(model_settings.get("enabled", True)):
                    comparison_records.append(
                        {
                            "horizon_hours": horizon,
                            "model": model_name,
                            "feature_set": feature_set_name,
                            "status": "disabled",
                        }
                    )
                    continue

                try:
                    estimator = _make_estimator(
                        model_name,
                        model_settings,
                        horizon_hours=horizon,
                        random_state=random_state,
                    )

                    sampled_train = _sample_training_rows(
                        train,
                        maximum_rows=maximum_training_rows,
                        random_state=random_state,
                    )
                    started = time.perf_counter()
                    estimator.fit(
                        sampled_train[selected_features],
                        sampled_train[target_column],
                    )
                    training_seconds = time.perf_counter() - started

                    validation_prediction = estimator.predict(validation[selected_features])
                    validation_metrics = calculate_regression_metrics(
                        validation[target_column],
                        validation_prediction,
                    )

                    complexity_order = {
                        "persistence": 0,
                        "recent_trend": 1,
                        "ridge": 2,
                        "random_forest": 3,
                        "xgboost": 4,
                        "lightgbm": 5,
                    }[model_name]
                    selection_key = (
                        validation_metrics["mae"],
                        validation_metrics["median_absolute_error"],
                        abs(validation_metrics["bias"]),
                        complexity_order,
                    )
                    candidates.append(
                        (
                            selection_key,
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
                            "validation_rows": len(validation),
                            "training_seconds": training_seconds,
                            **{
                                f"validation_{key}": value
                                for key, value in validation_metrics.items()
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
            raise RuntimeError(f"No forecasting candidate completed at {horizon}h.")

        (
            _,
            selected_model_name,
            selected_feature_set,
            selected_features,
            selected_estimator,
            validation_metrics,
        ) = min(candidates, key=lambda item: item[0])

        validation_prediction = selected_estimator.predict(validation[selected_features])
        test_prediction = selected_estimator.predict(test[selected_features])
        test_metrics = calculate_regression_metrics(
            test[target_column],
            test_prediction,
        )

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
                    CURRENT_WEIGHT_FEATURE,
                    target_column_name,
                ]
            ].copy()
            result = result.rename(
                columns={
                    CURRENT_WEIGHT_FEATURE: ("current_weight_kg"),
                    target_column_name: "actual_delta_kg",
                }
            )
            result["predicted_delta_kg"] = prediction
            result["actual_future_weight_kg"] = (
                result["current_weight_kg"] + result["actual_delta_kg"]
            )
            result["predicted_future_weight_kg"] = (
                result["current_weight_kg"] + result["predicted_delta_kg"]
            )
            result["error_kg"] = result["predicted_delta_kg"] - result["actual_delta_kg"]
            return result

        validation_predictions = build_predictions(
            validation,
            validation_prediction,
            target_column_name=target_column,
        )
        test_predictions = build_predictions(
            test,
            test_prediction,
            target_column_name=target_column,
        )

        validation_predictions.to_parquet(
            output_directory / f"selected_validation_predictions_{horizon}h.parquet",
            index=False,
        )
        test_predictions.to_parquet(
            output_directory / f"selected_test_predictions_{horizon}h.parquet",
            index=False,
        )

        all_per_hive.append(
            _per_hive_metrics(
                validation_predictions,
                horizon_hours=horizon,
                split="validation",
                model_name=selected_model_name,
                feature_set_name=selected_feature_set,
            )
        )
        all_per_hive.append(
            _per_hive_metrics(
                test_predictions,
                horizon_hours=horizon,
                split="test",
                model_name=selected_model_name,
                feature_set_name=selected_feature_set,
            )
        )

        joblib.dump(
            selected_estimator,
            model_directory / f"selected_weight_forecaster_{horizon}h.joblib",
        )
        _write_json(
            model_directory / f"selected_weight_forecaster_{horizon}h.json",
            {
                "horizon_hours": horizon,
                "model": selected_model_name,
                "feature_set": selected_feature_set,
                "feature_columns": selected_features,
                "target": target_column,
                "probability_status": ("not_applicable_regression_forecast"),
            },
        )

        summary_by_horizon[str(horizon)] = {
            "selected_model": selected_model_name,
            "selected_feature_set": selected_feature_set,
            "selected_feature_count": len(selected_features),
            "training_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "validation": validation_metrics,
            "test": test_metrics,
        }

    comparison = pd.DataFrame(comparison_records)
    comparison.to_csv(
        output_directory / "weight_forecasting_comparison.csv",
        index=False,
    )
    if all_per_hive:
        pd.concat(
            all_per_hive,
            ignore_index=True,
        ).to_csv(
            output_directory / "selected_forecaster_per_hive_metrics.csv",
            index=False,
        )

    target_audit = {
        "source_feature_rows": len(feature_rows),
        "horizons_hours": horizons,
        "available_rows_by_horizon_and_split": (
            dataset.melt(
                id_vars=[SPLIT_COLUMN],
                value_vars=[f"weight_delta_next_{horizon}h_kg" for horizon in horizons],
                var_name="target",
                value_name="value",
            )
            .loc[lambda frame: frame["value"].notna()]
            .groupby([SPLIT_COLUMN, "target"])
            .size()
            .rename("rows")
            .reset_index()
            .to_dict(orient="records")
        ),
        "target_policy": (
            "Future weight is joined at the exact requested timestamp "
            "within the same hive and official split. Rows without an "
            "exact endpoint or with a split crossing are excluded."
        ),
    }
    _write_json(
        output_directory / "weight_forecasting_target_audit.json",
        target_audit,
    )

    summary = {
        "research_stage": ("Label-independent future hive-weight forecasting."),
        "selection_rule": (
            "For each horizon, minimize validation MAE, then median "
            "absolute error, absolute bias and model complexity."
        ),
        "horizons": summary_by_horizon,
        "warnings": [
            (
                "Forecast accuracy does not by itself prove honey "
                "maturity or optimal harvesting time."
            ),
            ("Humidity is generated; the no-humidity feature set must be reported separately."),
            (
                "Readiness scoring must be transparent and validated "
                "prospectively with beekeeper-confirmed harvest data."
            ),
        ],
    }
    _write_json(
        output_directory / "weight_forecasting_summary.json",
        summary,
    )

    return {
        "status": "completed",
        "horizons": summary_by_horizon,
        "comparison_path": str(output_directory / "weight_forecasting_comparison.csv"),
        "summary_path": str(output_directory / "weight_forecasting_summary.json"),
        "target_audit_path": str(output_directory / "weight_forecasting_target_audit.json"),
    }
