from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .audit import binary_target_persistence_audit, feature_leakage_audit
from .calibration import calibrate_component_weights
from .config import PATHS
from .features import (
    FEATURE_SCHEMA_VERSION,
    MINIMUM_TRAINING_HISTORY_HOURS,
    SENSORS,
    TARGET_COLUMN,
    build_supervised_dataset,
    normalise_historical,
    target_columns,
)
from .scoring import (
    CODE_TO_LEVEL,
    HEALTH_LEVEL_ORDER,
    BroodHealthScoreConfig,
    health_level_code,
    score_definition,
)

try:
    from xgboost import XGBRegressor
except ImportError:  # Optional dependency.
    XGBRegressor = None

ProgressCallback = Callable[[str, dict[str, Any]], None]


def _notify(callback: ProgressCallback | None, event: str, **payload: Any) -> None:
    if callback:
        callback(event, payload)


def _load_frame(path: Path | None = None) -> pd.DataFrame:
    source = Path(path or PATHS.clean_data)
    if source.exists():
        if source.suffix.lower() in {".xlsx", ".xls"}:
            return normalise_historical(
                pd.read_excel(source, sheet_name="Common_Dataset")
            )
        if source.suffix.lower() == ".csv":
            return normalise_historical(pd.read_csv(source))
        try:
            return normalise_historical(pd.read_parquet(source))
        except (ImportError, ValueError):
            pass
    if PATHS.raw_workbook.exists():
        return normalise_historical(
            pd.read_excel(PATHS.raw_workbook, sheet_name="Common_Dataset")
        )
    raise FileNotFoundError("No common brood-health training dataset was found.")


def _assign_hive_splits(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
) -> dict[str, str]:
    """Assign complete hives to train/validation/test before score calibration."""

    audit = frame[["hive_id"]].copy()
    if TARGET_COLUMN in frame.columns:
        audit["observed"] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    else:
        audit["observed"] = np.nan

    hive_stats = (
        audit.groupby("hive_id", observed=True)
        .agg(
            observed_rate=("observed", "mean"),
            rows=("hive_id", "size"),
        )
        .reset_index()
    )
    hive_stats["observed_rate"] = hive_stats["observed_rate"].fillna(0.5)
    if len(hive_stats) < 5:
        raise ValueError("At least five hives are required for held-out evaluation")

    quantiles = min(5, max(2, int(hive_stats["observed_rate"].nunique())))
    try:
        hive_stats["stratum"] = pd.qcut(
            hive_stats["observed_rate"], q=quantiles, duplicates="drop"
        )
    except ValueError:
        hive_stats["stratum"] = "all"

    rng = np.random.default_rng(random_state)
    assignments: dict[str, str] = {}
    pattern = ("test", "validation", "train", "train", "train")
    for stratum_index, (_, group) in enumerate(
        hive_stats.groupby("stratum", observed=True, sort=False)
    ):
        hives = group["hive_id"].astype(str).tolist()
        rng.shuffle(hives)
        for position, hive_id in enumerate(hives):
            assignments[hive_id] = pattern[(position + stratum_index) % len(pattern)]

    for required in ("train", "validation", "test"):
        if required not in assignments.values():
            donor = max(
                ("train", "validation", "test"),
                key=lambda name: list(assignments.values()).count(name),
            )
            candidate = next(
                hive for hive, split in assignments.items() if split == donor
            )
            assignments[candidate] = required
    return assignments


def _candidate_models(*, fast_mode: bool) -> dict[str, Pipeline]:
    tree_count = 24 if fast_mode else 220
    hgb_iterations = 90 if fast_mode else 360
    models: dict[str, Pipeline] = {
        "Dummy Median": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", DummyRegressor(strategy="median")),
            ]
        ),
        "Ridge Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=15.0)),
            ]
        ),
        "Histogram Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    MultiOutputRegressor(
                        HistGradientBoostingRegressor(
                            learning_rate=0.05,
                            max_iter=hgb_iterations,
                            max_leaf_nodes=19,
                            min_samples_leaf=60,
                            l2_regularization=2.0,
                            random_state=42,
                        ),
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=tree_count,
                        max_depth=14,
                        min_samples_leaf=10,
                        max_features=0.50,
                        bootstrap=True,
                        max_samples=0.45 if fast_mode else 0.70,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    ExtraTreesRegressor(
                        n_estimators=tree_count,
                        max_depth=16,
                        min_samples_leaf=8,
                        max_features=0.55,
                        bootstrap=True,
                        max_samples=0.45 if fast_mode else 0.70,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    if XGBRegressor is not None and not fast_mode:
        models["XGBoost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    MultiOutputRegressor(
                        XGBRegressor(
                            n_estimators=420,
                            max_depth=6,
                            learning_rate=0.04,
                            subsample=0.80,
                            colsample_bytree=0.75,
                            objective="reg:squarederror",
                            eval_metric="mae",
                            random_state=42,
                            n_jobs=-1,
                            tree_method="hist",
                        ),
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return models


def _systematic_cap(
    x: pd.DataFrame,
    y: pd.DataFrame,
    metadata: pd.DataFrame,
    maximum: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(x) <= maximum:
        return (
            x.reset_index(drop=True),
            y.reset_index(drop=True),
            metadata.reset_index(drop=True),
        )

    selected: list[int] = []
    proportions = metadata["hive_id"].value_counts(normalize=True)
    for hive_id, proportion in proportions.items():
        positions = np.flatnonzero(metadata["hive_id"].eq(hive_id).to_numpy())
        quota = max(30, round(maximum * float(proportion)))
        if len(positions) > quota:
            positions = positions[
                np.linspace(0, len(positions) - 1, quota, dtype=int)
            ]
        selected.extend(positions.tolist())
    selected = sorted(set(selected))
    if len(selected) > maximum:
        selected = (
            np.asarray(selected)[
                np.linspace(0, len(selected) - 1, maximum, dtype=int)
            ]
            .astype(int)
            .tolist()
        )
    return (
        x.iloc[selected].reset_index(drop=True),
        y.iloc[selected].reset_index(drop=True),
        metadata.iloc[selected].reset_index(drop=True),
    )


def _clip_predictions(values: np.ndarray, horizon: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = np.repeat(array[:, None], horizon, axis=1)
    return np.clip(array, 1.0, 100.0)


def _classification_metrics(
    actual_score: np.ndarray,
    predicted_score: np.ndarray,
) -> dict[str, Any]:
    actual_level = health_level_code(actual_score)
    predicted_level = health_level_code(predicted_score)
    critical_true = actual_level == 0
    critical_predicted = predicted_level == 0
    return {
        "health_level_accuracy": float(
            accuracy_score(actual_level, predicted_level)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual_level, predicted_level)
        ),
        "macro_f1": float(
            f1_score(
                actual_level,
                predicted_level,
                average="macro",
                zero_division=0,
            )
        ),
        "critical_recall": float(
            recall_score(critical_true, critical_predicted, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            actual_level,
            predicted_level,
            labels=[0, 1, 2, 3],
        )
        .astype(int)
        .tolist(),
        "level_labels": list(HEALTH_LEVEL_ORDER),
    }


def _score_metrics(
    y_true: pd.DataFrame,
    predicted: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    horizon = y_true.shape[1]
    actual_matrix = np.clip(y_true.to_numpy(dtype=float), 1.0, 100.0)
    predicted_matrix = _clip_predictions(predicted, horizon)

    exact_actual = actual_matrix[:, -1]
    exact_predicted = predicted_matrix[:, -1]
    minimum_actual = actual_matrix.min(axis=1)
    minimum_predicted = predicted_matrix.min(axis=1)
    current = metadata["current_score"].to_numpy(dtype=float)

    exact = {
        "mae": float(mean_absolute_error(exact_actual, exact_predicted)),
        "mse": float(mean_squared_error(exact_actual, exact_predicted)),
        "rmse": float(
            math.sqrt(mean_squared_error(exact_actual, exact_predicted))
        ),
        "r2": float(r2_score(exact_actual, exact_predicted)),
        **_classification_metrics(exact_actual, exact_predicted),
    }
    safety = {
        "mae": float(mean_absolute_error(minimum_actual, minimum_predicted)),
        "mse": float(mean_squared_error(minimum_actual, minimum_predicted)),
        "rmse": float(
            math.sqrt(mean_squared_error(minimum_actual, minimum_predicted))
        ),
        "r2": float(r2_score(minimum_actual, minimum_predicted)),
        **_classification_metrics(minimum_actual, minimum_predicted),
    }

    per_horizon: list[dict[str, Any]] = []
    for index in range(horizon):
        actual = actual_matrix[:, index]
        forecast = predicted_matrix[:, index]
        per_horizon.append(
            {
                "horizon_hours": index + 1,
                "mae": float(mean_absolute_error(actual, forecast)),
                "mse": float(mean_squared_error(actual, forecast)),
                "rmse": float(
                    math.sqrt(mean_squared_error(actual, forecast))
                ),
                "r2": float(r2_score(actual, forecast)),
                "health_level_accuracy": float(
                    accuracy_score(
                        health_level_code(actual),
                        health_level_code(forecast),
                    )
                ),
            }
        )

    transition_mask = metadata["transition_window"].fillna(False).to_numpy(bool)
    deterioration_true = metadata["deterioration_event"].fillna(False).to_numpy(bool)
    predicted_deterioration = (
        (health_level_code(exact_predicted) < health_level_code(current))
        | ((current - exact_predicted) >= 10.0)
    )

    transition_metrics: dict[str, Any]
    if transition_mask.any():
        transition_actual = exact_actual[transition_mask]
        transition_predicted = exact_predicted[transition_mask]
        transition_metrics = {
            "rows": int(transition_mask.sum()),
            "mae": float(
                mean_absolute_error(transition_actual, transition_predicted)
            ),
            "rmse": float(
                math.sqrt(
                    mean_squared_error(transition_actual, transition_predicted)
                )
            ),
            **_classification_metrics(
                transition_actual,
                transition_predicted,
            ),
        }
    else:
        transition_metrics = {
            "rows": 0,
            "mae": None,
            "rmse": None,
            "health_level_accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "critical_recall": None,
            "confusion_matrix": [],
            "level_labels": list(HEALTH_LEVEL_ORDER),
        }

    deterioration = {
        "events": int(deterioration_true.sum()),
        "recall": float(
            recall_score(
                deterioration_true,
                predicted_deterioration,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                deterioration_true,
                predicted_deterioration,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                deterioration_true,
                predicted_deterioration,
                zero_division=0,
            )
        ),
    }

    return {
        "multi_horizon_mae": float(
            mean_absolute_error(actual_matrix, predicted_matrix)
        ),
        "exact_horizon": exact,
        "safety_minimum": safety,
        "transition": transition_metrics,
        "deterioration": deterioration,
        "per_horizon": per_horizon,
        # Compatibility aliases used by the current frontend and report.
        "test_mae": exact["mae"],
        "test_mse": exact["mse"],
        "test_rmse": exact["rmse"],
        "test_r2": exact["r2"],
        "health_level_accuracy": exact["health_level_accuracy"],
        "balanced_accuracy": exact["balanced_accuracy"],
        "macro_f1": exact["macro_f1"],
        "critical_recall": exact["critical_recall"],
        "transition_mae": transition_metrics["mae"],
        "transition_rmse": transition_metrics["rmse"],
        "transition_level_accuracy": transition_metrics[
            "health_level_accuracy"
        ],
        "transition_critical_recall": transition_metrics["critical_recall"],
        "confusion_matrix": exact["confusion_matrix"],
        "level_labels": exact["level_labels"],
    }


def _persistence_predictions(
    metadata: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    current = metadata["current_score"].to_numpy(dtype=float)
    return np.repeat(current[:, None], horizon, axis=1)


def _group_cv_exact_mae(
    estimator: Pipeline,
    x: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series,
    *,
    fast_mode: bool,
) -> tuple[float | None, float | None, int]:
    unique_groups = int(pd.Series(groups).nunique())
    folds = min(2 if fast_mode else 3, unique_groups)
    if folds < 2:
        return None, None, 0

    maximum = 8_000 if fast_mode else 50_000
    meta = pd.DataFrame({"hive_id": groups.astype(str).to_numpy()})
    x_cv, y_cv, meta_cv = _systematic_cap(x, y, meta, maximum)
    splitter = GroupKFold(n_splits=folds)
    errors: list[float] = []
    for train_positions, test_positions in splitter.split(
        x_cv,
        y_cv,
        groups=meta_cv["hive_id"],
    ):
        fold_model = clone(estimator)
        fold_model.fit(x_cv.iloc[train_positions], y_cv.iloc[train_positions])
        forecast = _clip_predictions(
            fold_model.predict(x_cv.iloc[test_positions]),
            y_cv.shape[1],
        )
        errors.append(
            float(
                mean_absolute_error(
                    y_cv.iloc[test_positions, -1],
                    forecast[:, -1],
                )
            )
        )
    return float(np.mean(errors)), float(np.std(errors, ddof=0)), folds


def _selection_key(
    validation: dict[str, Any],
    persistence: dict[str, Any],
) -> tuple[float, float, float, float, float, float]:
    exact = validation["exact_horizon"]
    transition = validation["transition"]
    beats_persistence = float(
        exact["mae"] < persistence["exact_horizon"]["mae"]
    )
    transition_accuracy = float(transition.get("health_level_accuracy") or 0.0)
    deterioration_recall = float(validation["deterioration"]["recall"])
    critical_recall = float(exact["critical_recall"])
    return (
        beats_persistence,
        transition_accuracy,
        deterioration_recall,
        critical_recall,
        -float(exact["mae"]),
        -float(validation["multi_horizon_mae"]),
    )


def _feature_group(feature: str) -> str:
    if feature.startswith("temperature"):
        return "temperature"
    if feature.startswith("humidity"):
        return "humidity"
    if feature.startswith("co2"):
        return "co2"
    if feature.startswith("weight"):
        return "weight_stability"
    return "time_and_interactions"


def _feature_importance(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    regressor = model.named_steps["regressor"]
    values: np.ndarray | None = None

    if hasattr(regressor, "feature_importances_"):
        values = np.asarray(regressor.feature_importances_, dtype=float)
    elif hasattr(regressor, "coef_"):
        coefficients = np.asarray(regressor.coef_, dtype=float)
        values = np.abs(coefficients).mean(axis=0)
    elif hasattr(regressor, "estimators_"):
        child_values = []
        for estimator in regressor.estimators_:
            if hasattr(estimator, "feature_importances_"):
                child_values.append(
                    np.asarray(estimator.feature_importances_, dtype=float)
                )
            elif hasattr(estimator, "coef_"):
                child_values.append(
                    np.abs(np.asarray(estimator.coef_, dtype=float)).reshape(-1)
                )
        if child_values:
            values = np.vstack(child_values).mean(axis=0)

    if values is None or len(values) != len(feature_columns):
        size = min(400, len(x_test))
        positions = np.linspace(0, len(x_test) - 1, size, dtype=int)
        permutation = permutation_importance(
            model,
            x_test.iloc[positions],
            y_test.iloc[positions],
            scoring="neg_mean_absolute_error",
            n_repeats=2,
            random_state=42,
            n_jobs=1,
        )
        values = np.asarray(permutation.importances_mean, dtype=float)

    importance = pd.DataFrame(
        {"feature": feature_columns, "importance": np.clip(values, 0.0, None)}
    )
    total = float(importance["importance"].sum())
    importance["importance_percentage"] = (
        importance["importance"] / total * 100.0 if total > 0 else 0.0
    )
    importance["sensor_group"] = importance["feature"].map(_feature_group)
    return importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


def _training_reference(
    frame: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    reference: dict[str, dict[str, float]] = {}
    for sensor in ("temperature_c", "humidity_pct", "co2_ppm", "weight_kg"):
        values = pd.to_numeric(frame[sensor], errors="coerce").dropna()
        reference[sensor] = {
            "p01": float(values.quantile(0.01)),
            "p05": float(values.quantile(0.05)),
            "median": float(values.median()),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
        }

    ordered = frame.sort_values(["hive_id", "timestamp"])
    weight = pd.to_numeric(ordered["weight_kg"], errors="coerce")
    previous = weight.groupby(ordered["hive_id"], sort=False).shift(24)
    change = ((weight - previous) / previous.abs().clip(lower=1.0)) * 100.0
    change = change.dropna()
    reference["weight_change_pct_24h"] = {
        "p01": float(change.quantile(0.01)),
        "p05": float(change.quantile(0.05)),
        "median": float(change.median()),
        "p95": float(change.quantile(0.95)),
        "p99": float(change.quantile(0.99)),
    }
    return reference


def _save_reports(
    summary: dict[str, Any],
    y_test: pd.DataFrame,
    predicted: np.ndarray,
    metadata: pd.DataFrame,
    importance: pd.DataFrame,
) -> list[dict[str, str]]:
    directory = PATHS.report_dir / "model"
    directory.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, str]] = []

    def save(fig: plt.Figure, filename: str, title: str) -> None:
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        images.append(
            {
                "filename": filename,
                "title": title,
                "url": f"/api/brood-health/reports/{filename}",
            }
        )

    successful = [
        row for row in summary["all_models"] if row.get("status") == "ok"
    ]
    names = [row["model"] for row in successful]
    exact_mae = [row["test"]["exact_horizon"]["mae"] for row in successful]
    transition_accuracy = [
        100.0
        * float(
            row["test"]["transition"].get("health_level_accuracy") or 0.0
        )
        for row in successful
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, exact_mae)
    ax.set_ylabel("Exact +6 h MAE (score points)")
    ax.set_title("Model comparison: exact forecast error")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "model_exact_mae_comparison.png", "Exact-horizon MAE comparison")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, transition_accuracy)
    ax.set_ylabel("Transition level accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model comparison during health transitions")
    ax.tick_params(axis="x", rotation=25)
    save(
        fig,
        "model_transition_accuracy.png",
        "Transition-level accuracy comparison",
    )

    actual = y_test.iloc[:, -1].to_numpy(dtype=float)
    forecast = _clip_predictions(predicted, y_test.shape[1])[:, -1]
    positions = np.linspace(0, len(actual) - 1, min(5_000, len(actual)), dtype=int)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual[positions], forecast[positions], s=9, alpha=0.25)
    ax.plot([1, 100], [1, 100], linestyle="--")
    ax.set_xlim(1, 100)
    ax.set_ylim(1, 100)
    ax.set_xlabel("Actual score at +6 h")
    ax.set_ylabel("Predicted score at +6 h")
    ax.set_title("Actual versus predicted exact forecast")
    save(
        fig,
        "actual_vs_predicted_exact_6h.png",
        "Actual versus predicted exact +6 h score",
    )

    matrix = np.asarray(summary["best_metrics"]["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix)
    ax.set_xticks(range(4), HEALTH_LEVEL_ORDER, rotation=20)
    ax.set_yticks(range(4), HEALTH_LEVEL_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Exact +6 h health-level confusion matrix")
    for row in range(4):
        for column in range(4):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    save(
        fig,
        "exact_6h_confusion_matrix.png",
        "Exact +6 h health-level confusion matrix",
    )

    top = importance.head(20).sort_values("importance_percentage")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance_percentage"])
    ax.set_xlabel("Relative importance (%)")
    ax.set_title("Top causal predictive features")
    save(fig, "feature_importance_v4.png", "Feature importance")

    per_horizon = pd.DataFrame(summary["best_metrics"]["per_horizon"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(per_horizon["horizon_hours"], per_horizon["mae"], marker="o")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE (score points)")
    ax.set_title("Forecast error across the 1–6 hour trajectory")
    save(fig, "horizon_mae_curve.png", "Forecast error by horizon")

    return images


def _serialisable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serialisable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialisable(item) for item in value]
    if isinstance(value, tuple):
        return [_serialisable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def run_training(
    *,
    data_path: Path | None = None,
    manifest_path: Path | None = None,  # Retained for API compatibility.
    horizon_hours: int = 6,
    fast_mode: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    horizon = int(horizon_hours)
    if not 1 <= horizon <= 24:
        raise ValueError("horizon_hours must be between 1 and 24")

    for directory in (
        PATHS.model_dir,
        PATHS.metrics_dir,
        PATHS.report_dir / "model",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    _notify(
        progress_callback,
        "loading",
        progress=3,
        message="Loading the common cleaned historical data",
    )
    frame = _load_frame(data_path)
    hive_assignments = _assign_hive_splits(frame)
    training_hives = {
        hive for hive, split in hive_assignments.items() if split == "train"
    }

    _notify(
        progress_callback,
        "weight_calibration",
        progress=7,
        message="Calibrating score weights on training hives only",
    )
    calibration = calibrate_component_weights(
        frame,
        training_hives=training_hives,
    )
    score_config = calibration.config
    if not calibration.comparison.empty:
        calibration.comparison.to_csv(PATHS.weight_sensitivity, index=False)

    binary_target_audit = binary_target_persistence_audit(
        frame,
        horizons=(1, horizon, 24),
    )

    _notify(
        progress_callback,
        "features",
        progress=11,
        message=(
            "Building causal features and exact 1–6 hour future score targets"
        ),
    )
    x, y, metadata, feature_columns = build_supervised_dataset(
        frame,
        horizon_hours=horizon,
        score_config=score_config,
    )
    metadata["split"] = (
        metadata["hive_id"].astype(str).map(hive_assignments).astype("string")
    )
    schema_audit = feature_leakage_audit(feature_columns)
    if not schema_audit["passed"]:
        raise RuntimeError(f"Feature-schema leakage audit failed: {schema_audit}")

    train_mask = metadata["split"].eq("train")
    validation_mask = metadata["split"].eq("validation")
    test_mask = metadata["split"].eq("test")
    x_train, y_train, meta_train = (
        x.loc[train_mask],
        y.loc[train_mask],
        metadata.loc[train_mask],
    )
    x_validation, y_validation, meta_validation = (
        x.loc[validation_mask],
        y.loc[validation_mask],
        metadata.loc[validation_mask],
    )
    x_test, y_test, meta_test = (
        x.loc[test_mask],
        y.loc[test_mask],
        metadata.loc[test_mask],
    )
    if min(len(x_train), len(x_validation), len(x_test)) == 0:
        raise ValueError("Train, validation or test partition is empty")

    limits = (
        (18_000, 7_000, 7_000)
        if fast_mode
        else (160_000, 55_000, 55_000)
    )
    x_train, y_train, meta_train = _systematic_cap(
        x_train,
        y_train,
        meta_train,
        limits[0],
    )
    x_validation, y_validation, meta_validation = _systematic_cap(
        x_validation,
        y_validation,
        meta_validation,
        limits[1],
    )
    x_test, y_test, meta_test = _systematic_cap(
        x_test,
        y_test,
        meta_test,
        limits[2],
    )

    persistence_validation = _score_metrics(
        y_validation,
        _persistence_predictions(meta_validation, horizon),
        meta_validation,
    )
    persistence_test = _score_metrics(
        y_test,
        _persistence_predictions(meta_test, horizon),
        meta_test,
    )

    candidates = _candidate_models(fast_mode=fast_mode)
    comparison: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for index, (name, estimator) in enumerate(candidates.items()):
        started = time.perf_counter()
        progress = 15 + int(index / max(len(candidates), 1) * 65)
        _notify(
            progress_callback,
            "model_start",
            progress=progress,
            model=name,
            message=f"Training {name}",
        )
        try:
            model = clone(estimator)
            model.fit(x_train, y_train)
            validation_prediction = _clip_predictions(
                model.predict(x_validation),
                horizon,
            )
            validation_metrics = _score_metrics(
                y_validation,
                validation_prediction,
                meta_validation,
            )
            test_prediction = _clip_predictions(model.predict(x_test), horizon)
            test_metrics = _score_metrics(
                y_test,
                test_prediction,
                meta_test,
            )

            cv_mean, cv_std, cv_folds = _group_cv_exact_mae(
                estimator,
                x_train,
                y_train,
                meta_train["hive_id"],
                fast_mode=fast_mode,
            )
            for metrics in (validation_metrics, test_metrics):
                metrics["cv_mae_mean"] = cv_mean
                metrics["cv_mae_std"] = cv_std
                metrics["cv_folds"] = cv_folds

            result = {
                "model": name,
                "status": "ok",
                "validation": validation_metrics,
                "test": test_metrics,
                "fit_seconds": float(time.perf_counter() - started),
            }
            comparison.append(result)
            fitted[name] = model
            test_predictions[name] = test_prediction
            _notify(
                progress_callback,
                "model_end",
                progress=min(progress + 10, 86),
                model=name,
                message=(
                    f"{name}: exact +{horizon} h MAE "
                    f"{test_metrics['exact_horizon']['mae']:.2f}; transition accuracy "
                    f"{100 * float(test_metrics['transition'].get('health_level_accuracy') or 0):.2f}%"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            comparison.append(
                {"model": name, "status": "failed", "error": str(exc)}
            )
            _notify(
                progress_callback,
                "model_end",
                progress=min(progress + 10, 86),
                model=name,
                message=f"{name} failed: {exc}",
            )

    successful = [
        item
        for item in comparison
        if item.get("status") == "ok" and item["model"] != "Dummy Median"
    ]
    if not successful:
        raise RuntimeError("Every non-baseline brood-health regressor failed")

    best_result = max(
        successful,
        key=lambda item: _selection_key(
            item["validation"],
            persistence_validation,
        ),
    )
    best_name = best_result["model"]
    evaluation_model = fitted[best_name]
    evaluation_prediction = test_predictions[best_name]

    _notify(
        progress_callback,
        "importance",
        progress=88,
        message="Calculating selected-model feature importance",
    )
    importance = _feature_importance(
        evaluation_model,
        x_test,
        y_test,
        feature_columns,
    )
    importance.to_csv(PATHS.feature_importance, index=False)

    # Residual interval is estimated only on validation hives.
    validation_prediction = _clip_predictions(
        fitted[best_name].predict(x_validation),
        horizon,
    )
    validation_exact_residual = np.abs(
        y_validation.iloc[:, -1].to_numpy(dtype=float)
        - validation_prediction[:, -1]
    )
    interval_80 = float(np.quantile(validation_exact_residual, 0.80))
    interval_90 = float(np.quantile(validation_exact_residual, 0.90))

    deployment_model = clone(candidates[best_name])
    deployment_model.fit(
        pd.concat([x_train, x_validation], ignore_index=True),
        pd.concat([y_train, y_validation], ignore_index=True),
    )

    reference_frame = frame.loc[
        frame["hive_id"].astype(str).isin(
            {
                hive
                for hive, split in hive_assignments.items()
                if split in {"train", "validation"}
            }
        )
    ]
    bundle = {
        "model": deployment_model,
        "model_name": best_name,
        "feature_columns": feature_columns,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "horizon_hours": horizon,
        "target_columns": target_columns(horizon),
        "primary_target": f"brood_health_score_t_plus_{horizon}h",
        "target_kind": "multi_horizon_exact_score_trajectory",
        "target_range": [1.0, 100.0],
        "score_config": score_config.to_dict(),
        "level_order": list(HEALTH_LEVEL_ORDER),
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "training_sensor_reference": _training_reference(reference_frame),
        "prediction_inputs": list(SENSORS),
        "prediction_interval_absolute_error": {
            "80_percent": interval_80,
            "90_percent": interval_90,
        },
        "weight_calibration": calibration.method,
        "weight_transfer_strategy": "relative_change_and_stability_only",
        "model_limitations": [
            "The 1–100 Brood Health Score is a transparent sensor-derived research index, not a direct veterinary measurement.",
            f"The primary forecast is the exact score at +{horizon} hours. The minimum across the predicted trajectory is a secondary safety indicator.",
            "Historical evaluation holds out complete hives, but Sri Lankan field performance must be verified against physical brood inspections.",
            "No accuracy value is capped or deliberately reduced. Transition and deterioration metrics must be interpreted alongside overall accuracy.",
            "External temperature and humidity are shown as context unless equivalent historical training variables are available.",
        ],
    }
    joblib.dump(bundle, PATHS.model_bundle)

    exact_actual = y_test.iloc[:, -1].to_numpy(dtype=float)
    exact_predicted = evaluation_prediction[:, -1]
    safety_actual = y_test.min(axis=1).to_numpy(dtype=float)
    safety_predicted = evaluation_prediction.min(axis=1)
    prediction_table = meta_test[
        [
            "hive_id",
            "timestamp",
            "target_timestamp",
            "current_score",
            "transition_window",
            "deterioration_event",
        ]
    ].copy()
    prediction_table["actual_exact_score"] = exact_actual
    prediction_table["predicted_exact_score"] = exact_predicted
    prediction_table["actual_safety_minimum"] = safety_actual
    prediction_table["predicted_safety_minimum"] = safety_predicted
    prediction_table["actual_level"] = [
        CODE_TO_LEVEL[int(code)] for code in health_level_code(exact_actual)
    ]
    prediction_table["predicted_level"] = [
        CODE_TO_LEVEL[int(code)] for code in health_level_code(exact_predicted)
    ]
    prediction_table.to_csv(PATHS.test_predictions, index=False)

    comparison_rows = []
    for item in comparison:
        if item.get("status") != "ok":
            comparison_rows.append(
                {"model": item["model"], "status": "failed", "error": item["error"]}
            )
            continue
        exact = item["test"]["exact_horizon"]
        transition = item["test"]["transition"]
        comparison_rows.append(
            {
                "model": item["model"],
                "status": "ok",
                "mae": exact["mae"],
                "mse": exact["mse"],
                "rmse": exact["rmse"],
                "r2": exact["r2"],
                "health_level_accuracy": exact["health_level_accuracy"],
                "critical_recall": exact["critical_recall"],
                "transition_level_accuracy": transition.get(
                    "health_level_accuracy"
                ),
                "transition_mae": transition.get("mae"),
                "deterioration_recall": item["test"]["deterioration"]["recall"],
                "cv_mae_mean": item["test"].get("cv_mae_mean"),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(PATHS.model_comparison, index=False)

    grouped_importance = (
        importance.groupby("sensor_group", observed=True)[
            "importance_percentage"
        ]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .to_dict(orient="records")
    )

    sample_positions = np.linspace(
        0,
        len(y_test) - 1,
        min(500, len(y_test)),
        dtype=int,
    )
    prediction_sample = [
        {
            "actual": float(exact_actual[position]),
            "predicted": float(exact_predicted[position]),
            "safety_actual": float(safety_actual[position]),
            "safety_predicted": float(safety_predicted[position]),
            "transition": bool(meta_test.iloc[position]["transition_window"]),
        }
        for position in sample_positions
    ]

    best_metrics = best_result["test"]
    summary: dict[str, Any] = {
        "version": "4.0",
        "trained": True,
        "best_model": best_name,
        "horizon_hours": horizon,
        "primary_target": f"brood_health_score_t_plus_{horizon}h",
        "primary_target_description": (
            f"Brood Health Score exactly {horizon} hours after the latest observation."
        ),
        "secondary_safety_target": f"minimum_predicted_score_within_next_{horizon}h",
        "secondary_target_description": (
            f"Minimum of the model's predicted 1–{horizon} hour trajectory; used only as a safety early-warning indicator."
        ),
        "forecast_strategy": {
            "type": "multi-horizon direct regression",
            "horizons": list(range(1, horizon + 1)),
            "primary_output": f"exact +{horizon} hour score",
            "secondary_output": f"minimum predicted score within 1–{horizon} hours",
            "reason": (
                "This provides the exact requested future score while retaining a worst-case "
                "trajectory indicator for earlier intervention."
            ),
        },
        "score_definition": score_definition(score_config),
        "weight_calibration": calibration.method,
        "weight_sensitivity_top": (
            calibration.comparison.head(20).to_dict(orient="records")
            if not calibration.comparison.empty
            else []
        ),
        "selection_rule": (
            "Select using validation hives only: first prefer models that beat current-score "
            "persistence on exact-horizon MAE, then maximize transition accuracy, deterioration "
            "recall and Critical recall, and minimize exact and multi-horizon MAE."
        ),
        "best_metrics": best_metrics,
        "best_validation_metrics": best_result["validation"],
        "all_models": comparison,
        "persistence_baseline": persistence_test,
        "binary_target_audit": binary_target_audit,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "top_features": importance.head(30).to_dict(orient="records"),
        "grouped_feature_importance": grouped_importance,
        "split_summary": {
            "strategy": "complete-hive 60/20/20 group holdout",
            "train_rows": len(x_train),
            "validation_rows": len(x_validation),
            "test_rows": len(x_test),
            "train_hives": int(meta_train["hive_id"].nunique()),
            "validation_hives": int(meta_validation["hive_id"].nunique()),
            "test_hives": int(meta_test["hive_id"].nunique()),
            "train_hive_ids": sorted(
                meta_train["hive_id"].astype(str).unique().tolist()
            ),
            "validation_hive_ids": sorted(
                meta_validation["hive_id"].astype(str).unique().tolist()
            ),
            "test_hive_ids": sorted(
                meta_test["hive_id"].astype(str).unique().tolist()
            ),
            "minimum_history_hours": MINIMUM_TRAINING_HISTORY_HOURS,
            "future_target_uses_only_later_rows": True,
            "score_weights_calibrated_on_training_hives_only": True,
        },
        "accuracy_interpretation": {
            "overall_level_accuracy_percent": float(
                100.0 * best_metrics["exact_horizon"]["health_level_accuracy"]
            ),
            "transition_level_accuracy_percent": float(
                100.0
                * float(
                    best_metrics["transition"].get("health_level_accuracy")
                    or 0.0
                )
            ),
            "deterioration_recall_percent": float(
                100.0 * best_metrics["deterioration"]["recall"]
            ),
            "persistence_exact_mae": float(
                persistence_test["exact_horizon"]["mae"]
            ),
            "model_exact_mae": float(best_metrics["exact_horizon"]["mae"]),
            "explanation": (
                "Overall accuracy may be high because most hours are stable. Transition-level "
                "accuracy and deterioration recall are the primary early-warning evidence. "
                "The implementation never forces accuracy into a desired range."
            ),
        },
        "leakage_audit": {
            **schema_audit,
            "rolling_features_shifted_before_aggregation": True,
            "whole_hives_held_out": True,
            "test_hives_used_for_model_selection": False,
            "observed_health_label_used_as_model_feature": False,
            "absolute_hive_weight_used_as_model_feature": False,
            "absolute_date_used_as_model_feature": False,
        },
        "prediction_interval": {
            "method": "absolute residual quantiles on validation hives",
            "80_percent_half_width": interval_80,
            "90_percent_half_width": interval_90,
        },
        "metrics_note": (
            "MAE, MSE, RMSE, R², health-level accuracy, Critical recall and group-CV "
            "MAE are reported for the exact +6-hour score. Transition and deterioration "
            "metrics are added because stable observations can inflate overall accuracy."
        ),
        "prediction_sample": prediction_sample,
        "trained_at_utc": bundle["trained_at_utc"],
        "model_limitations": bundle["model_limitations"],
    }
    summary["generated_images"] = _save_reports(
        summary,
        y_test,
        evaluation_prediction,
        meta_test,
        importance,
    )
    summary = _serialisable(summary)
    PATHS.training_summary.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _notify(
        progress_callback,
        "complete",
        progress=100,
        message=f"Training complete. Best model: {best_name}",
    )
    return summary
