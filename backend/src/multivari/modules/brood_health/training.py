from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
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
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .audit import binary_target_persistence_audit, feature_leakage_audit
from .config import PATHS
from .features import (
    FEATURE_SCHEMA_VERSION,
    SENSORS,
    TARGET_COLUMN,
    build_supervised_dataset,
    normalise_historical,
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
            return normalise_historical(pd.read_excel(source, sheet_name="Common_Dataset"))
        if source.suffix.lower() == ".csv":
            return normalise_historical(pd.read_csv(source))
        try:
            return normalise_historical(pd.read_parquet(source))
        except ImportError:
            pass
    if PATHS.raw_workbook.exists():
        return normalise_historical(pd.read_excel(PATHS.raw_workbook, sheet_name="Common_Dataset"))
    raise FileNotFoundError("No common brood-health training dataset was found.")


def _assign_hive_splits(metadata: pd.DataFrame, target: pd.Series, *, random_state: int = 42) -> pd.Series:
    """Create an unseen-hive 60/20/20 split stratified by hive score profile.

    The deployment hives in Sri Lanka are not the historical hives. Holding out whole
    hives is therefore a more honest primary evaluation than placing neighbouring rows
    from every hive in both training and testing.
    """

    audit = metadata[["hive_id"]].copy()
    audit["target_score"] = target.to_numpy()
    hive_stats = (
        audit.groupby("hive_id", observed=True)
        .agg(mean_score=("target_score", "mean"), critical_rate=("target_score", lambda x: float((x < 40).mean())), rows=("target_score", "size"))
        .reset_index()
    )
    if len(hive_stats) < 5:
        raise ValueError("At least five hives are required for group-held-out evaluation")

    quantiles = min(5, max(2, int(hive_stats["mean_score"].nunique())))
    try:
        hive_stats["stratum"] = pd.qcut(hive_stats["mean_score"], q=quantiles, duplicates="drop")
    except ValueError:
        hive_stats["stratum"] = "all"

    rng = np.random.default_rng(random_state)
    assignments: dict[str, str] = {}
    pattern = ("test", "validation", "train", "train", "train")
    for stratum_index, (_, group) in enumerate(hive_stats.groupby("stratum", observed=True, sort=False)):
        hives = group["hive_id"].astype(str).tolist()
        rng.shuffle(hives)
        for position, hive_id in enumerate(hives):
            assignments[hive_id] = pattern[(position + stratum_index) % len(pattern)]

    # Repair tiny edge cases without moving rows between hives.
    for required_split in ("train", "validation", "test"):
        if required_split not in assignments.values():
            donor = max(
                (name for name in ("train", "validation", "test") if list(assignments.values()).count(name) > 1),
                key=lambda name: list(assignments.values()).count(name),
            )
            candidate = next(hive for hive, split in assignments.items() if split == donor)
            assignments[candidate] = required_split

    return metadata["hive_id"].astype(str).map(assignments).astype("string")


def _candidate_models(*, fast_mode: bool) -> dict[str, Pipeline]:
    tree_count = 12 if fast_mode else 180
    models: dict[str, Pipeline] = {
        "Dummy Median": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("regressor", DummyRegressor(strategy="median"))]
        ),
        "Ridge Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=12.0)),
            ]
        ),
        "Histogram Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    HistGradientBoostingRegressor(
                        learning_rate=0.06,
                        max_iter=70 if fast_mode else 320,
                        max_leaf_nodes=19,
                        min_samples_leaf=45,
                        l2_regularization=1.0,
                        random_state=42,
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
                        min_samples_leaf=8,
                        max_features=0.55,
                        bootstrap=True,
                        max_samples=0.40 if fast_mode else 0.68,
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
                        max_depth=18,
                        min_samples_leaf=6,
                        max_features=0.60,
                        bootstrap=True,
                        max_samples=0.40 if fast_mode else 0.68,
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
                    XGBRegressor(
                        n_estimators=420,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.80,
                        colsample_bytree=0.75,
                        objective="reg:squarederror",
                        eval_metric="mae",
                        random_state=42,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        )
    return models


def _systematic_cap(
    x: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    maximum: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if len(x) <= maximum:
        return x.reset_index(drop=True), y.reset_index(drop=True), metadata.reset_index(drop=True)
    # Preserve every hive and the temporal ordering within it.
    selected: list[int] = []
    proportions = metadata["hive_id"].value_counts(normalize=True)
    for hive_id, proportion in proportions.items():
        positions = np.flatnonzero(metadata["hive_id"].eq(hive_id).to_numpy())
        quota = max(20, round(maximum * float(proportion)))
        if len(positions) > quota:
            positions = positions[np.linspace(0, len(positions) - 1, quota, dtype=int)]
        selected.extend(positions.tolist())
    selected = sorted(set(selected))
    if len(selected) > maximum:
        selected = np.asarray(selected)[np.linspace(0, len(selected) - 1, maximum, dtype=int)].tolist()
    return (
        x.iloc[selected].reset_index(drop=True),
        y.iloc[selected].reset_index(drop=True),
        metadata.iloc[selected].reset_index(drop=True),
    )


def _clip_scores(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1.0, 100.0)


def _safe_roc_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, score))


def _safe_average_precision(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(average_precision_score(y_true, score))


def _score_metrics(
    y_true: pd.Series,
    predicted_score: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    actual = _clip_scores(y_true)
    predicted = _clip_scores(predicted_score)
    actual_level = health_level_code(actual)
    predicted_level = health_level_code(predicted)

    critical_true = actual_level == 0
    critical_predicted = predicted_level == 0
    transition_mask = metadata["transition_window"].fillna(False).to_numpy(dtype=bool)
    observed_healthy = metadata["future_observed_healthy"].to_numpy(dtype=int)
    predicted_observed_healthy = (predicted >= 60.0).astype(int)

    result: dict[str, Any] = {
        "test_mae": float(mean_absolute_error(actual, predicted)),
        "test_rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "test_r2": float(r2_score(actual, predicted)),
        "health_level_accuracy": float(accuracy_score(actual_level, predicted_level)),
        "accuracy": float(accuracy_score(actual_level, predicted_level)),
        "balanced_accuracy": float(balanced_accuracy_score(actual_level, predicted_level)),
        "macro_f1": float(f1_score(actual_level, predicted_level, average="macro", zero_division=0)),
        "critical_recall": float(recall_score(critical_true, critical_predicted, zero_division=0)),
        "unhealthy_recall": float(
            recall_score(observed_healthy == 0, predicted_observed_healthy == 0, zero_division=0)
        ),
        "observed_binary_accuracy": float(accuracy_score(observed_healthy, predicted_observed_healthy)),
        "observed_binary_balanced_accuracy": float(
            balanced_accuracy_score(observed_healthy, predicted_observed_healthy)
        ),
        "roc_auc": _safe_roc_auc(observed_healthy, predicted / 100.0),
        "pr_auc": _safe_average_precision(observed_healthy, predicted / 100.0),
        "confusion_matrix": confusion_matrix(actual_level, predicted_level, labels=[0, 1, 2, 3]).astype(int).tolist(),
        "level_labels": list(HEALTH_LEVEL_ORDER),
        "actual_score_mean": float(np.mean(actual)),
        "predicted_score_mean": float(np.mean(predicted)),
    }

    if transition_mask.any():
        transition_actual = actual[transition_mask]
        transition_predicted = predicted[transition_mask]
        transition_actual_level = actual_level[transition_mask]
        transition_predicted_level = predicted_level[transition_mask]
        transition_critical_true = transition_actual_level == 0
        transition_critical_predicted = transition_predicted_level == 0
        result.update(
            {
                "transition_rows": int(transition_mask.sum()),
                "transition_mae": float(mean_absolute_error(transition_actual, transition_predicted)),
                "transition_rmse": float(math.sqrt(mean_squared_error(transition_actual, transition_predicted))),
                "transition_level_accuracy": float(
                    accuracy_score(transition_actual_level, transition_predicted_level)
                ),
                "transition_balanced_accuracy": float(
                    balanced_accuracy_score(transition_actual_level, transition_predicted_level)
                ),
                "transition_critical_recall": float(
                    recall_score(transition_critical_true, transition_critical_predicted, zero_division=0)
                ),
            }
        )
    else:
        result.update(
            {
                "transition_rows": 0,
                "transition_mae": None,
                "transition_rmse": None,
                "transition_level_accuracy": None,
                "transition_balanced_accuracy": None,
                "transition_critical_recall": None,
            }
        )
    return result


def _persistence_metrics(y_true: pd.Series, metadata: pd.DataFrame) -> dict[str, Any]:
    return _score_metrics(y_true, metadata["current_score"].to_numpy(), metadata)


def _group_cv_mae(
    estimator: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    fast_mode: bool,
) -> tuple[float | None, float | None, int]:
    unique_groups = pd.Series(groups).nunique()
    splits = min(2 if fast_mode else 3, int(unique_groups))
    if splits < 2:
        return None, None, 0

    maximum = 8_000 if fast_mode else 60_000
    metadata = pd.DataFrame({"hive_id": groups.astype(str).to_numpy()})
    x_cv, y_cv, meta_cv = _systematic_cap(x, y, metadata, maximum)
    splitter = GroupKFold(n_splits=splits)
    errors: list[float] = []
    for train_positions, test_positions in splitter.split(x_cv, y_cv, groups=meta_cv["hive_id"]):
        fold_model = clone(estimator)
        fold_model.fit(x_cv.iloc[train_positions], y_cv.iloc[train_positions])
        predicted = _clip_scores(fold_model.predict(x_cv.iloc[test_positions]))
        errors.append(float(mean_absolute_error(y_cv.iloc[test_positions], predicted)))
    return float(np.mean(errors)), float(np.std(errors, ddof=0)), splits


def _selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    critical_gate = 1.0 if float(metrics.get("transition_critical_recall") or 0.0) >= 0.70 else 0.0
    transition_accuracy = float(metrics.get("transition_level_accuracy") or 0.0)
    transition_mae = float(metrics.get("transition_mae") or 999.0)
    return (
        critical_gate,
        transition_accuracy,
        -transition_mae,
        -float(metrics["test_mae"]),
        float(metrics["test_r2"]),
    )


def _feature_group(feature: str) -> str:
    if feature.startswith("temperature"):
        return "temperature_c"
    if feature.startswith("humidity"):
        return "humidity_pct"
    if feature.startswith("co2"):
        return "co2_ppm"
    if feature.startswith("weight"):
        return "weight_kg"
    return "time_and_interactions"


def _feature_importance(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list[str],
) -> pd.DataFrame:
    regressor = model.named_steps["regressor"]
    if hasattr(regressor, "feature_importances_"):
        values = np.asarray(regressor.feature_importances_, dtype=float)
    elif hasattr(regressor, "coef_"):
        values = np.abs(np.asarray(regressor.coef_)).reshape(-1)
    else:
        size = min(350, len(x_test))
        positions = np.linspace(0, len(x_test) - 1, size, dtype=int)
        permutation = permutation_importance(
            model,
            x_test.iloc[positions],
            y_test.iloc[positions],
            scoring="neg_mean_absolute_error",
            n_repeats=1,
            random_state=42,
            n_jobs=1,
        )
        values = np.asarray(permutation.importances_mean, dtype=float)
    importance = pd.DataFrame({"feature": feature_columns, "importance": values})
    importance["importance"] = importance["importance"].clip(lower=0.0)
    total = float(importance["importance"].sum())
    importance["importance_percentage"] = (
        importance["importance"] / total * 100.0 if total > 0 else 0.0
    )
    importance["sensor_group"] = importance["feature"].map(_feature_group)
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def _sensor_reference(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    reference: dict[str, dict[str, float]] = {}
    for sensor in SENSORS:
        values = pd.to_numeric(frame[sensor], errors="coerce").dropna()
        reference[sensor] = {
            "minimum": float(values.min()),
            "p01": float(values.quantile(0.01)),
            "p05": float(values.quantile(0.05)),
            "median": float(values.median()),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
            "maximum": float(values.max()),
        }
    return reference


def _save_training_reports(
    summary: dict[str, Any],
    y_test: pd.Series,
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
        images.append({"filename": filename, "title": title, "url": f"/api/brood-health/reports/{filename}"})

    models = [item for item in summary["all_models"] if item.get("status") == "ok"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh([item["model"] for item in models], [item["test"]["test_mae"] for item in models])
    ax.set_xlabel("Unseen-hive test MAE (score points; lower is better)")
    ax.set_title("Brood Health Score model comparison")
    ax.invert_yaxis()
    save(fig, "model_mae_comparison.png", "Model MAE comparison")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(
        [item["model"] for item in models],
        [100.0 * float(item["test"].get("transition_level_accuracy") or 0.0) for item in models],
    )
    ax.set_xlabel("Transition-window level accuracy (%)")
    ax.set_title("Early-warning accuracy around deterioration and level changes")
    ax.invert_yaxis()
    save(fig, "transition_accuracy_comparison.png", "Transition-window accuracy")

    matrix = np.asarray(summary["best_metrics"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix)
    ax.set_xticks(range(4), HEALTH_LEVEL_ORDER, rotation=20)
    ax.set_yticks(range(4), HEALTH_LEVEL_ORDER)
    ax.set_xlabel("Predicted future level")
    ax.set_ylabel("Actual future level")
    ax.set_title("Unseen-hive four-level confusion matrix")
    threshold = matrix.max() * 0.55 if matrix.size else 0
    for row in range(4):
        for column in range(4):
            ax.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.colorbar(image, ax=ax)
    save(fig, "score_level_confusion_matrix.png", "Four-level confusion matrix")

    actual = _clip_scores(y_test)
    predicted_clipped = _clip_scores(predicted)
    sample_positions = np.linspace(0, len(actual) - 1, min(5_000, len(actual)), dtype=int)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(actual[sample_positions], predicted_clipped[sample_positions], s=8, alpha=0.25)
    ax.plot([1, 100], [1, 100], linestyle="--")
    ax.set_xlim(1, 100)
    ax.set_ylim(1, 100)
    ax.set_xlabel("Actual future minimum score")
    ax.set_ylabel("Predicted future minimum score")
    ax.set_title("Actual versus predicted Brood Health Score")
    save(fig, "actual_vs_predicted_score.png", "Actual versus predicted score")

    residual = predicted_clipped - actual
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residual, bins=60)
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Prediction error (predicted − actual score)")
    ax.set_ylabel("Rows")
    ax.set_title("Unseen-hive test residual distribution")
    save(fig, "score_residual_distribution.png", "Residual distribution")

    top = importance.head(20).sort_values("importance_percentage")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance_percentage"])
    ax.set_xlabel("Relative importance (%)")
    ax.set_title("Top causal predictive features")
    save(fig, "feature_importance.png", "Feature importance")

    transition = metadata["transition_window"].to_numpy(dtype=bool)
    if transition.any():
        fig, ax = plt.subplots(figsize=(6.5, 6))
        positions = np.flatnonzero(transition)
        if len(positions) > 4_000:
            positions = positions[np.linspace(0, len(positions) - 1, 4_000, dtype=int)]
        ax.scatter(actual[positions], predicted_clipped[positions], s=10, alpha=0.30)
        ax.plot([1, 100], [1, 100], linestyle="--")
        ax.set_xlim(1, 100)
        ax.set_ylim(1, 100)
        ax.set_xlabel("Actual score during transition windows")
        ax.set_ylabel("Predicted score")
        ax.set_title("Transition-window predictions")
        save(fig, "transition_window_predictions.png", "Transition-window predictions")

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
    manifest_path: Path | None = None,  # Kept for API compatibility; group holdout is primary.
    horizon_hours: int = 6,
    fast_mode: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    horizon = int(horizon_hours)
    if not 1 <= horizon <= 168:
        raise ValueError("horizon_hours must be between 1 and 168")
    for directory in (PATHS.model_dir, PATHS.metrics_dir, PATHS.report_dir / "model"):
        directory.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, "loading", progress=3, message="Loading common cleaned historical data")
    frame = _load_frame(data_path)
    training_reference = _sensor_reference(frame)
    binary_target_audit = binary_target_persistence_audit(frame, horizons=(1, horizon, 24))

    _notify(
        progress_callback,
        "module_preprocessing",
        progress=7,
        message="Applying brood-specific range checks and causal short-gap handling",
    )
    _notify(
        progress_callback,
        "features",
        progress=10,
        message="Building past-only sensor, lag, rolling, trend and interaction features",
    )
    x, y, metadata, feature_columns = build_supervised_dataset(frame, horizon_hours=horizon)
    metadata["split"] = _assign_hive_splits(metadata, y)
    schema_audit = feature_leakage_audit(feature_columns)
    if not schema_audit["passed"]:
        raise RuntimeError(f"Feature-schema leakage audit failed: {schema_audit}")

    train_mask = metadata["split"].eq("train")
    validation_mask = metadata["split"].eq("validation")
    test_mask = metadata["split"].eq("test")
    x_train, y_train, meta_train = x.loc[train_mask], y.loc[train_mask], metadata.loc[train_mask]
    x_validation, y_validation, meta_validation = (
        x.loc[validation_mask],
        y.loc[validation_mask],
        metadata.loc[validation_mask],
    )
    x_test, y_test, meta_test = x.loc[test_mask], y.loc[test_mask], metadata.loc[test_mask]
    if min(len(x_train), len(x_validation), len(x_test)) == 0:
        raise ValueError("The group-held-out split produced an empty train, validation or test partition")

    limits = (20_000, 8_000, 8_000) if fast_mode else (170_000, 60_000, 60_000)
    x_train, y_train, meta_train = _systematic_cap(x_train, y_train, meta_train, limits[0])
    x_validation, y_validation, meta_validation = _systematic_cap(
        x_validation, y_validation, meta_validation, limits[1]
    )
    x_test, y_test, meta_test = _systematic_cap(x_test, y_test, meta_test, limits[2])

    persistence_validation = _persistence_metrics(y_validation, meta_validation)
    persistence_test = _persistence_metrics(y_test, meta_test)

    candidates = _candidate_models(fast_mode=fast_mode)
    comparison: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for index, (name, estimator) in enumerate(candidates.items()):
        started = time.perf_counter()
        progress = 15 + int(index / max(len(candidates), 1) * 67)
        _notify(progress_callback, "model_start", progress=progress, model=name, message=f"Training {name}")
        try:
            model = clone(estimator)
            model.fit(x_train, y_train)
            validation_prediction = _clip_scores(model.predict(x_validation))
            validation_metrics = _score_metrics(y_validation, validation_prediction, meta_validation)
            test_prediction = _clip_scores(model.predict(x_test))
            test_metrics = _score_metrics(y_test, test_prediction, meta_test)

            cv_mean, cv_std, cv_folds = _group_cv_mae(
                estimator,
                x_train,
                y_train,
                meta_train["hive_id"],
                fast_mode=fast_mode,
            )
            test_metrics["cv_mae_mean"] = cv_mean
            test_metrics["cv_mae_std"] = cv_std
            test_metrics["cv_folds"] = cv_folds
            validation_metrics["cv_mae_mean"] = cv_mean
            validation_metrics["cv_mae_std"] = cv_std
            validation_metrics["cv_folds"] = cv_folds

            test_metrics["mae_improvement_over_persistence"] = (
                persistence_test["test_mae"] - test_metrics["test_mae"]
            )
            test_metrics["transition_mae_improvement_over_persistence"] = (
                float(persistence_test.get("transition_mae") or 0.0)
                - float(test_metrics.get("transition_mae") or 0.0)
            )
            validation_metrics["mae_improvement_over_persistence"] = (
                persistence_validation["test_mae"] - validation_metrics["test_mae"]
            )
            validation_metrics["transition_mae_improvement_over_persistence"] = (
                float(persistence_validation.get("transition_mae") or 0.0)
                - float(validation_metrics.get("transition_mae") or 0.0)
            )

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
                progress=min(progress + 10, 87),
                model=name,
                message=(
                    f"{name}: test MAE {test_metrics['test_mae']:.2f}; "
                    f"transition level accuracy {100 * float(test_metrics.get('transition_level_accuracy') or 0):.2f}%"
                ),
            )
        # Model comparison is fault-tolerant: one estimator failure must not abort the run.
        except Exception as exc:  # noqa: BLE001
            comparison.append({"model": name, "status": "failed", "error": str(exc)})
            _notify(
                progress_callback,
                "model_end",
                progress=min(progress + 10, 87),
                model=name,
                message=f"{name} failed: {exc}",
            )

    successful = [item for item in comparison if item.get("status") == "ok" and item["model"] != "Dummy Median"]
    if not successful:
        raise RuntimeError("Every non-baseline brood-health regressor failed")
    best_result = max(successful, key=lambda item: _selection_key(item["validation"]))
    best_name = best_result["model"]
    evaluation_model = fitted[best_name]
    evaluation_prediction = test_predictions[best_name]

    _notify(progress_callback, "importance", progress=89, message="Calculating selected-model feature importance")
    importance = _feature_importance(evaluation_model, x_test, y_test, feature_columns)
    importance.to_csv(PATHS.feature_importance, index=False)

    # Refit the selected deployment model on every non-test hive after model selection.
    deployment_model = clone(candidates[best_name])
    x_deployment = pd.concat([x_train, x_validation], ignore_index=True)
    y_deployment = pd.concat([y_train, y_validation], ignore_index=True)
    deployment_model.fit(x_deployment, y_deployment)

    score_config = BroodHealthScoreConfig()
    bundle = {
        "model": deployment_model,
        "model_name": best_name,
        "feature_columns": feature_columns,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "horizon_hours": horizon,
        "target_column": "future_minimum_brood_health_score",
        "target_kind": "continuous_future_window_minimum_score",
        "target_range": [1.0, 100.0],
        "score_config": score_config.to_dict(),
        "level_order": list(HEALTH_LEVEL_ORDER),
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "training_sensor_reference": training_reference,
        "prediction_inputs": list(SENSORS),
        "model_limitations": [
            "The 1–100 Brood Health Score is a transparent sensor-derived research index, not a direct biological measurement.",
            "The model predicts the minimum score expected within the configured future window, not a guaranteed exact value at one timestamp.",
            "Unseen-hive historical evaluation is more realistic than a random row split but cannot guarantee Sri Lankan field performance.",
            "Live sensor calibration and distribution shift must be checked continuously.",
            "Physical brood inspection is required to confirm every Poor or Critical alert.",
        ],
    }
    joblib.dump(bundle, PATHS.model_bundle)

    prediction_table = meta_test[
        [
            "hive_id",
            "timestamp",
            "target_timestamp",
            "current_score",
            "future_observed_healthy",
            "transition_window",
        ]
    ].copy()
    prediction_table["actual_future_minimum_score"] = y_test.to_numpy()
    prediction_table["predicted_future_minimum_score"] = evaluation_prediction
    prediction_table["actual_level"] = [CODE_TO_LEVEL[int(code)] for code in health_level_code(y_test)]
    prediction_table["predicted_level"] = [CODE_TO_LEVEL[int(code)] for code in health_level_code(evaluation_prediction)]
    prediction_table.to_csv(PATHS.test_predictions, index=False)

    grouped_importance = (
        importance.groupby("sensor_group", observed=True)["importance_percentage"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .to_dict(orient="records")
    )

    observed_healthy = meta_test["future_observed_healthy"].to_numpy(dtype=int)
    probability_like_score = evaluation_prediction / 100.0
    if np.unique(observed_healthy).size >= 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(observed_healthy, probability_like_score)
        precision_curve, recall_curve, _ = precision_recall_curve(observed_healthy, probability_like_score)
        roc_positions = np.linspace(0, len(false_positive_rate) - 1, min(180, len(false_positive_rate)), dtype=int)
        pr_positions = np.linspace(0, len(precision_curve) - 1, min(180, len(precision_curve)), dtype=int)
        curves = {
            "roc": [
                {
                    "false_positive_rate": float(false_positive_rate[position]),
                    "true_positive_rate": float(true_positive_rate[position]),
                }
                for position in roc_positions
            ],
            "precision_recall": [
                {"recall": float(recall_curve[position]), "precision": float(precision_curve[position])}
                for position in pr_positions
            ],
        }
    else:
        curves = {"roc": [], "precision_recall": []}

    sample_positions = np.linspace(0, len(y_test) - 1, min(500, len(y_test)), dtype=int)
    prediction_sample = [
        {
            "actual": float(y_test.iloc[position]),
            "predicted": float(evaluation_prediction[position]),
            "transition": bool(meta_test.iloc[position]["transition_window"]),
        }
        for position in sample_positions
    ]

    best_metrics = best_result["test"]
    horizon_audit = next(
        (row for row in binary_target_audit["horizons"] if row["horizon_hours"] == horizon),
        None,
    )
    persistence_percent = (
        100.0 * float(horizon_audit["persistence_accuracy"])
        if horizon_audit and horizon_audit.get("persistence_accuracy") is not None
        else None
    )
    summary: dict[str, Any] = {
        "trained": True,
        "best_model": best_name,
        "horizon_hours": horizon,
        "target": "future_minimum_brood_health_score",
        "target_description": (
            f"Minimum sensor-derived Brood Health Score expected during the next {horizon} hours."
        ),
        "target_formulation": {
            "current_output": "Current Brood Health Score (1–100) calculated transparently from live sensor conditions.",
            "future_output": f"Predicted minimum Brood Health Score within the next {horizon} hours.",
            "why_changed": (
                f"At the {horizon}-hour horizon, the historical binary status remains unchanged for "
                f"{persistence_percent:.2f}% of comparable rows. Very high binary accuracy therefore mostly "
                "measured persistence rather than useful advance warning."
                if persistence_percent is not None
                else "The previous nearby-future binary target was dominated by status persistence rather than useful advance warning."
            ),
            "observed_binary_label_role": (
                f"{TARGET_COLUMN} is retained for EDA and secondary biological alignment metrics, but is never an input feature."
            ),
            "score_definition": score_definition(score_config),
        },
        "selection_rule": (
            "Select on held-out validation hives: require transition critical recall of at least 0.70 when possible, "
            "then maximize transition-window level accuracy, minimize transition MAE and overall MAE. The test hives "
            "remain untouched until final reporting."
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
            "strategy": "stratified group holdout by hive",
            "train_rows": len(x_train),
            "validation_rows": len(x_validation),
            "test_rows": len(x_test),
            "train_hives": int(meta_train["hive_id"].nunique()),
            "validation_hives": int(meta_validation["hive_id"].nunique()),
            "test_hives": int(meta_test["hive_id"].nunique()),
            "train_hive_ids": sorted(meta_train["hive_id"].astype(str).unique().tolist()),
            "validation_hive_ids": sorted(meta_validation["hive_id"].astype(str).unique().tolist()),
            "test_hive_ids": sorted(meta_test["hive_id"].astype(str).unique().tolist()),
            "minimum_history_hours": 72,
            "future_target_uses_only_later_rows": True,
        },
        "accuracy_interpretation": {
            "primary_early_warning_metric": "transition_level_accuracy",
            "primary_early_warning_accuracy_percent": float(
                100.0 * float(best_metrics.get("transition_level_accuracy") or 0.0)
            ),
            "overall_level_accuracy_percent": float(100.0 * best_metrics["health_level_accuracy"]),
            "persistence_overall_level_accuracy_percent": float(
                100.0 * persistence_test["health_level_accuracy"]
            ),
            "persistence_transition_level_accuracy_percent": float(
                100.0 * float(persistence_test.get("transition_level_accuracy") or 0.0)
            ),
            "explanation": (
                "Overall level accuracy can remain above 90% because most hive hours are stable. The transition-window "
                "metric measures the difficult rows where the score drops by at least 10 points or crosses a health level, "
                "and is the more relevant early-warning accuracy. No metric is capped or deliberately degraded."
            ),
        },
        "leakage_audit": {
            **schema_audit,
            "rolling_features_shifted_before_aggregation": True,
            "whole_hives_held_out": True,
            "test_hives_used_for_model_selection": False,
        },
        "metrics_note": (
            "MAE, RMSE, R², health-level accuracy, critical recall and group-CV MAE are the requested model-comparison "
            "metrics. Transition MAE and transition-level accuracy are added because stable rows otherwise make accuracy misleading."
        ),
        "curves": curves,
        "prediction_sample": prediction_sample,
        "trained_at_utc": bundle["trained_at_utc"],
        "model_limitations": bundle["model_limitations"],
    }
    summary["generated_images"] = _save_training_reports(
        summary,
        y_test,
        evaluation_prediction,
        meta_test,
        importance,
    )
    summary = _serialisable(summary)
    PATHS.training_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _notify(progress_callback, "complete", progress=100, message=f"Training complete. Best model: {best_name}")
    return summary