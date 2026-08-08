from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from multivari.modules.harvesting.research_model_comparison import (
    HIVE_COLUMN,
    SPLIT_COLUMN,
    TIMESTAMP_COLUMN,
    attach_future_event_metadata,
    calculate_session_balanced_weights,
    cluster_harvest_sessions,
    sample_training_rows,
)


class Calibrator(Protocol):
    method_name: str

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform raw classifier scores into calibrated probabilities."""


@dataclass
class IdentityCalibrator:
    method_name: str = "identity"

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return np.asarray(probabilities, dtype=float)


@dataclass
class PlattCalibrator:
    estimator: LogisticRegression
    epsilon: float
    method_name: str = "platt"

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        logits = _probability_logit(probabilities, epsilon=self.epsilon)
        return self.estimator.predict_proba(logits.reshape(-1, 1))[:, 1]


@dataclass
class IsotonicCalibrator:
    estimator: IsotonicRegression
    epsilon: float
    method_name: str = "isotonic"

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated = self.estimator.predict(np.asarray(probabilities, dtype=float))
        return np.clip(calibrated, self.epsilon, 1.0 - self.epsilon)


def _resolve_path(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
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


def _probability_logit(
    probabilities: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probabilities, dtype=float),
        epsilon,
        1.0 - epsilon,
    )
    return np.log(clipped / (1.0 - clipped))


def _make_estimator(
    model_name: str,
    settings: dict[str, Any],
    *,
    random_state: int,
) -> BaseEstimator:
    if model_name == "logistic_regression":
        model = LogisticRegression(
            C=float(settings["C"]),
            penalty="elasticnet",
            l1_ratio=float(settings["l1_ratio"]),
            solver="saga",
            max_iter=int(settings["max_iter"]),
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", model),
            ]
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
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
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost") from error

        return XGBClassifier(
            n_estimators=int(settings["n_estimators"]),
            learning_rate=float(settings["learning_rate"]),
            max_depth=int(settings["max_depth"]),
            min_child_weight=float(settings["min_child_weight"]),
            subsample=float(settings["subsample"]),
            colsample_bytree=float(settings["colsample_bytree"]),
            reg_alpha=float(settings["reg_alpha"]),
            reg_lambda=float(settings["reg_lambda"]),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm") from error

        return LGBMClassifier(
            n_estimators=int(settings["n_estimators"]),
            learning_rate=float(settings["learning_rate"]),
            num_leaves=int(settings["num_leaves"]),
            max_depth=int(settings["max_depth"]),
            min_child_samples=int(settings["min_child_samples"]),
            subsample=float(settings["subsample"]),
            colsample_bytree=float(settings["colsample_bytree"]),
            reg_alpha=float(settings["reg_alpha"]),
            reg_lambda=float(settings["reg_lambda"]),
            objective="binary",
            n_jobs=-1,
            random_state=random_state,
            verbosity=-1,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def _fit_estimator(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    weights: np.ndarray,
) -> None:
    if isinstance(estimator, Pipeline):
        estimator.fit(
            features,
            target,
            model__sample_weight=weights,
        )
    else:
        estimator.fit(
            features,
            target,
            sample_weight=weights,
        )


def _positive_probabilities(
    estimator: BaseEstimator,
    features: pd.DataFrame,
) -> np.ndarray:
    probabilities = estimator.predict_proba(features)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Classifier predict_proba must return two columns.")
    return probabilities[:, 1].astype(float)


def assign_grouped_hive_folds(
    rows: pd.DataFrame,
    *,
    target_column: str,
    requested_folds: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Assign whole hives to deterministic out-of-fold partitions.

    Positive hives are distributed first so every fold receives at least one
    positive hive. Negative-only hives are then balanced by row count.
    """
    if requested_folds < 2:
        raise ValueError("requested_folds must be at least two")

    _require_columns(
        rows,
        {HIVE_COLUMN, target_column},
        frame_name="Training rows",
    )

    statistics = (
        rows.groupby(HIVE_COLUMN, observed=True)
        .agg(
            rows=(target_column, "size"),
            positive_rows=(target_column, "sum"),
        )
        .reset_index()
    )
    positive_hives = statistics.loc[statistics["positive_rows"].gt(0)].copy()
    negative_hives = statistics.loc[statistics["positive_rows"].eq(0)].copy()

    if len(positive_hives) < 2:
        raise ValueError("Grouped calibration requires at least two positive hives.")

    fold_count = min(requested_folds, len(positive_hives))
    loads = [
        {
            "fold": fold,
            "positive_rows": 0,
            "rows": 0,
            "hives": 0,
        }
        for fold in range(fold_count)
    ]
    assignments: dict[str, int] = {}

    positive_hives = positive_hives.sort_values(
        ["positive_rows", "rows", HIVE_COLUMN],
        ascending=[False, False, True],
    )
    for record in positive_hives.itertuples(index=False):
        destination = min(
            loads,
            key=lambda item: (
                item["positive_rows"],
                item["rows"],
                item["hives"],
                item["fold"],
            ),
        )
        assignments[str(record.hive_id)] = int(destination["fold"])
        destination["positive_rows"] += int(record.positive_rows)
        destination["rows"] += int(record.rows)
        destination["hives"] += 1

    negative_hives = negative_hives.sort_values(
        ["rows", HIVE_COLUMN],
        ascending=[False, True],
    )
    for record in negative_hives.itertuples(index=False):
        destination = min(
            loads,
            key=lambda item: (
                item["rows"],
                item["hives"],
                item["fold"],
            ),
        )
        assignments[str(record.hive_id)] = int(destination["fold"])
        destination["rows"] += int(record.rows)
        destination["hives"] += 1

    fold_series = rows[HIVE_COLUMN].astype(str).map(assignments)
    if fold_series.isna().any():
        raise RuntimeError("Some hives were not assigned to an OOF fold.")

    audit = statistics.copy()
    audit["fold"] = audit[HIVE_COLUMN].astype(str).map(assignments).astype(int)
    audit = audit.sort_values(["fold", HIVE_COLUMN]).reset_index(drop=True)

    fold_summary = (
        audit.groupby("fold", observed=True)
        .agg(
            hives=(HIVE_COLUMN, "nunique"),
            rows=("rows", "sum"),
            positive_rows=("positive_rows", "sum"),
            positive_hives=(
                "positive_rows",
                lambda values: int((values > 0).sum()),
            ),
        )
        .reset_index()
    )
    if fold_summary["positive_hives"].lt(1).any():
        raise RuntimeError("Every OOF fold must contain at least one positive hive.")

    return fold_series.astype(int), audit


def build_grouped_oof_predictions(
    rows: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    model_settings: dict[str, Any],
    requested_folds: int,
    maximum_negative_to_positive_ratio: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(
        rows,
        {
            TIMESTAMP_COLUMN,
            HIVE_COLUMN,
            SPLIT_COLUMN,
            target_column,
            *feature_columns,
        },
        frame_name="Training rows",
    )

    fold_ids, fold_audit = assign_grouped_hive_folds(
        rows,
        target_column=target_column,
        requested_folds=requested_folds,
    )
    working = rows.copy()
    working["calibration_fold"] = fold_ids.to_numpy()

    predictions: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []

    for fold in sorted(working["calibration_fold"].unique()):
        fold_train = working.loc[working["calibration_fold"].ne(fold)].copy()
        fold_validation = working.loc[working["calibration_fold"].eq(fold)].copy()

        if fold_train[target_column].nunique() < 2:
            raise ValueError(f"OOF fold {fold} training data has only one class.")
        if fold_validation[target_column].nunique() < 2:
            raise ValueError(f"OOF fold {fold} validation data has only one class.")

        sampled = sample_training_rows(
            fold_train,
            target_column=target_column,
            maximum_negative_to_positive_ratio=(maximum_negative_to_positive_ratio),
            random_state=random_state + int(fold),
        )
        weights = calculate_session_balanced_weights(
            sampled,
            target_column=target_column,
        )
        estimator = _make_estimator(
            model_name,
            model_settings,
            random_state=random_state + int(fold),
        )

        started = time.perf_counter()
        _fit_estimator(
            estimator,
            sampled[feature_columns],
            sampled[target_column],
            weights,
        )
        elapsed = time.perf_counter() - started

        raw_probability = _positive_probabilities(
            estimator,
            fold_validation[feature_columns],
        )
        output = fold_validation[
            [
                TIMESTAMP_COLUMN,
                HIVE_COLUMN,
                SPLIT_COLUMN,
                target_column,
            ]
        ].copy()
        output["calibration_fold"] = int(fold)
        output["raw_probability"] = raw_probability
        predictions.append(output)

        fold_records.append(
            {
                "fold": int(fold),
                "training_hives": int(fold_train[HIVE_COLUMN].nunique()),
                "validation_hives": int(fold_validation[HIVE_COLUMN].nunique()),
                "training_rows_before_sampling": len(fold_train),
                "training_rows_after_sampling": len(sampled),
                "training_positive_rows": int(sampled[target_column].sum()),
                "validation_rows": len(fold_validation),
                "validation_positive_rows": int(fold_validation[target_column].sum()),
                "training_seconds": elapsed,
            }
        )

    result = pd.concat(predictions, ignore_index=True)
    if len(result) != len(rows):
        raise RuntimeError("OOF predictions do not cover every training row.")
    duplicated = result.duplicated(
        [TIMESTAMP_COLUMN, HIVE_COLUMN],
        keep=False,
    )
    if duplicated.any():
        raise RuntimeError("OOF predictions contain duplicate hive/timestamp rows.")

    fold_run_audit = pd.DataFrame(fold_records)
    fold_hive_audit = fold_audit.merge(
        fold_run_audit,
        on="fold",
        how="left",
        validate="many_to_one",
    )
    return (
        result.sort_values([TIMESTAMP_COLUMN, HIVE_COLUMN]).reset_index(drop=True),
        fold_hive_audit,
    )


def fit_calibrator(
    method: str,
    raw_probability: np.ndarray,
    target: np.ndarray,
    *,
    epsilon: float,
) -> Calibrator:
    raw = np.asarray(raw_probability, dtype=float)
    y_true = np.asarray(target, dtype=int)

    if y_true.min() == y_true.max():
        raise ValueError("Calibration target must contain both classes.")

    if method == "identity":
        return IdentityCalibrator()

    if method == "platt":
        logits = _probability_logit(raw, epsilon=epsilon)
        estimator = LogisticRegression(
            C=1_000_000.0,
            solver="lbfgs",
            max_iter=5000,
            random_state=0,
        )
        estimator.fit(logits.reshape(-1, 1), y_true)
        return PlattCalibrator(estimator=estimator, epsilon=epsilon)

    if method == "isotonic":
        estimator = IsotonicRegression(
            y_min=epsilon,
            y_max=1.0 - epsilon,
            out_of_bounds="clip",
        )
        estimator.fit(raw, y_true)
        return IsotonicCalibrator(estimator=estimator, epsilon=epsilon)

    raise ValueError(f"Unsupported calibration method: {method}")


def build_reliability_table(
    target: np.ndarray,
    probabilities: np.ndarray,
    *,
    requested_bins: int,
) -> pd.DataFrame:
    if requested_bins < 2:
        raise ValueError("requested_bins must be at least two")

    frame = pd.DataFrame(
        {
            "target": np.asarray(target, dtype=int),
            "probability": np.asarray(probabilities, dtype=float),
        }
    )
    unique_values = int(frame["probability"].nunique())
    bin_count = min(requested_bins, unique_values, len(frame))
    if bin_count < 2:
        frame["bin"] = 0
    else:
        ranked = frame["probability"].rank(
            method="first",
            pct=True,
        )
        frame["bin"] = np.minimum(
            (ranked * bin_count).astype(int),
            bin_count - 1,
        )

    reliability = (
        frame.groupby("bin", observed=True)
        .agg(
            rows=("target", "size"),
            positive_rows=("target", "sum"),
            mean_probability=("probability", "mean"),
            minimum_probability=("probability", "min"),
            maximum_probability=("probability", "max"),
            observed_event_rate=("target", "mean"),
        )
        .reset_index()
    )
    reliability["absolute_calibration_gap"] = (
        reliability["mean_probability"] - reliability["observed_event_rate"]
    ).abs()
    return reliability


def _calibration_slope_intercept(
    target: np.ndarray,
    probabilities: np.ndarray,
    *,
    epsilon: float,
) -> tuple[float | None, float | None]:
    y_true = np.asarray(target, dtype=int)
    if y_true.min() == y_true.max():
        return None, None

    logits = _probability_logit(probabilities, epsilon=epsilon)
    estimator = LogisticRegression(
        C=1_000_000.0,
        solver="lbfgs",
        max_iter=5000,
        random_state=0,
    )
    try:
        estimator.fit(logits.reshape(-1, 1), y_true)
    except (RuntimeError, ValueError):
        return None, None

    return (
        float(estimator.intercept_[0]),
        float(estimator.coef_[0, 0]),
    )


def calculate_calibration_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    *,
    reliability_bins: int,
    epsilon: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_true = np.asarray(target, dtype=int)
    y_prob = np.clip(
        np.asarray(probabilities, dtype=float),
        epsilon,
        1.0 - epsilon,
    )
    reliability = build_reliability_table(
        y_true,
        y_prob,
        requested_bins=reliability_bins,
    )
    total_rows = int(reliability["rows"].sum())
    weights = reliability["rows"] / total_rows
    ece = float((weights * reliability["absolute_calibration_gap"]).sum())
    mce = float(reliability["absolute_calibration_gap"].max())
    intercept, slope = _calibration_slope_intercept(
        y_true,
        y_prob,
        epsilon=epsilon,
    )

    metrics = {
        "rows": len(y_true),
        "positive_rows": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "mean_probability": float(y_prob.mean()),
        "minimum_probability": float(y_prob.min()),
        "median_probability": float(np.median(y_prob)),
        "maximum_probability": float(y_prob.max()),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
    return metrics, reliability


def select_calibration_method(
    comparison: pd.DataFrame,
) -> str:
    validation = comparison.loc[
        comparison["split"].eq("validation") & comparison["status"].eq("ok")
    ].copy()
    if validation.empty:
        raise RuntimeError("No calibration method has valid validation metrics.")

    complexity = {
        "identity": 0,
        "platt": 1,
        "isotonic": 2,
    }
    validation["complexity"] = validation["method"].map(complexity)
    ranked = validation.sort_values(
        [
            "brier_score",
            "log_loss",
            "expected_calibration_error",
            "complexity",
        ],
        ascending=True,
    )
    return str(ranked.iloc[0]["method"])


def evaluate_calibration_gate(
    comparison: pd.DataFrame,
    *,
    selected_method: str,
    minimum_validation_brier_improvement_fraction: float,
    require_validation_ece_not_worse: bool,
    maximum_test_brier_degradation_fraction: float,
    positive_hive_count: int,
    minimum_positive_hives: int,
) -> dict[str, Any]:
    indexed = comparison.set_index(["method", "split"])

    raw_validation = indexed.loc[("identity", "validation")]
    selected_validation = indexed.loc[(selected_method, "validation")]
    raw_test = indexed.loc[("identity", "test")]
    selected_test = indexed.loc[(selected_method, "test")]

    raw_validation_brier = float(raw_validation["brier_score"])
    selected_validation_brier = float(selected_validation["brier_score"])
    raw_test_brier = float(raw_test["brier_score"])
    selected_test_brier = float(selected_test["brier_score"])

    validation_improvement = (
        raw_validation_brier - selected_validation_brier
    ) / raw_validation_brier
    test_degradation = (selected_test_brier - raw_test_brier) / raw_test_brier

    validation_ece_not_worse = float(selected_validation["expected_calibration_error"]) <= float(
        raw_validation["expected_calibration_error"]
    )

    conditions = {
        "non_identity_method_selected": selected_method != "identity",
        "validation_brier_improvement_sufficient": (
            validation_improvement >= minimum_validation_brier_improvement_fraction
        ),
        "validation_ece_not_worse": (
            validation_ece_not_worse if require_validation_ece_not_worse else True
        ),
        "test_brier_not_materially_worse": (
            test_degradation <= maximum_test_brier_degradation_fraction
        ),
        "positive_hive_count_sufficient": (positive_hive_count >= minimum_positive_hives),
    }
    gate_passed = all(conditions.values())

    return {
        "status": (
            "research_calibration_gate_passed"
            if gate_passed
            else "research_calibration_gate_failed"
        ),
        "gate_passed": gate_passed,
        "selected_method": selected_method,
        "conditions": conditions,
        "validation_brier_improvement_fraction": (validation_improvement),
        "test_brier_degradation_fraction": test_degradation,
        "minimum_validation_brier_improvement_fraction": (
            minimum_validation_brier_improvement_fraction
        ),
        "maximum_test_brier_degradation_fraction": (maximum_test_brier_degradation_fraction),
        "positive_hive_count": positive_hive_count,
        "minimum_positive_hives": minimum_positive_hives,
        "operational_calibration_allowed": False,
        "operational_limitation": (
            "Validation has two reviewed events and test has one reviewed "
            "event. Passing this research gate would support only a "
            "provisional academic HUI, not operational calibration."
        ),
    }


def run_probability_calibration_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = config["probability_calibration"]
    model_settings_root = config["research_model_comparison"]

    feature_path = _resolve_path(
        root,
        settings["feature_dataset_path"],
    )
    event_path = _resolve_path(
        root,
        settings["reviewed_events_path"],
    )
    selected_model_path = _resolve_path(
        root,
        settings["selected_model_path"],
    )
    metadata_path = _resolve_path(
        root,
        settings["selected_model_metadata_path"],
    )
    feature_columns_path = _resolve_path(
        root,
        settings["selected_feature_columns_path"],
    )
    validation_path = _resolve_path(
        root,
        settings["validation_predictions_path"],
    )
    test_path = _resolve_path(
        root,
        settings["test_predictions_path"],
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

    target_column = str(settings["target_column"])
    horizon_hours = int(settings["horizon_hours"])
    session_gap_hours = int(settings["session_gap_hours"])
    requested_folds = int(settings["grouped_oof_folds"])
    random_state = int(settings["random_state"])
    negative_ratio = int(settings["maximum_negative_to_positive_ratio"])
    reliability_bins = int(settings["reliability_bins"])
    epsilon = float(settings["probability_clip_epsilon"])

    if not selected_model_path.exists():
        raise FileNotFoundError(f"Missing selected classifier: {selected_model_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_payload = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    selected_model_name = str(metadata["model_name"])
    feature_columns = [str(value) for value in feature_payload["features"]]
    selected_model_settings = model_settings_root["models"][selected_model_name]

    features = pd.read_parquet(feature_path)
    events = pd.read_parquet(event_path)
    validation_predictions = pd.read_parquet(validation_path)
    test_predictions = pd.read_parquet(test_path)

    features[TIMESTAMP_COLUMN] = pd.to_datetime(
        features[TIMESTAMP_COLUMN],
        errors="raise",
    )
    events["event_start"] = pd.to_datetime(
        events["event_start"],
        errors="raise",
    )
    for frame in (validation_predictions, test_predictions):
        frame[TIMESTAMP_COLUMN] = pd.to_datetime(
            frame[TIMESTAMP_COLUMN],
            errors="raise",
        )

    session_events = cluster_harvest_sessions(
        events,
        session_gap_hours=session_gap_hours,
    )
    rows = attach_future_event_metadata(
        features,
        session_events,
        target_column=target_column,
        horizon_hours=horizon_hours,
    )
    training_rows = rows.loc[rows[SPLIT_COLUMN].eq("train")].copy()

    oof_predictions, fold_audit = build_grouped_oof_predictions(
        training_rows,
        feature_columns=feature_columns,
        target_column=target_column,
        model_name=selected_model_name,
        model_settings=selected_model_settings,
        requested_folds=requested_folds,
        maximum_negative_to_positive_ratio=negative_ratio,
        random_state=random_state,
    )

    method_settings = settings["methods"]
    methods = [
        method
        for method in ("identity", "platt", "isotonic")
        if bool(method_settings.get(method, {}).get("enabled", True))
    ]
    if "identity" not in methods:
        methods.insert(0, "identity")

    split_frames = {
        "training_oof": oof_predictions,
        "validation": validation_predictions,
        "test": test_predictions,
    }
    calibrators: dict[str, Calibrator] = {}
    comparison_records: list[dict[str, Any]] = []
    reliability_outputs: list[pd.DataFrame] = []
    calibrated_by_method: dict[
        tuple[str, str],
        np.ndarray,
    ] = {}

    training_raw = oof_predictions["raw_probability"].to_numpy(dtype=float)
    training_target = oof_predictions[target_column].to_numpy(dtype=int)

    for method in methods:
        try:
            calibrator = fit_calibrator(
                method,
                training_raw,
                training_target,
                epsilon=epsilon,
            )
            calibrators[method] = calibrator

            for split_name, frame in split_frames.items():
                calibrated = calibrator.predict(frame["raw_probability"].to_numpy(dtype=float))
                calibrated = np.clip(
                    calibrated,
                    epsilon,
                    1.0 - epsilon,
                )
                calibrated_by_method[(method, split_name)] = calibrated
                metrics, reliability = calculate_calibration_metrics(
                    frame[target_column].to_numpy(dtype=int),
                    calibrated,
                    reliability_bins=reliability_bins,
                    epsilon=epsilon,
                )
                comparison_records.append(
                    {
                        "method": method,
                        "split": split_name,
                        "status": "ok",
                        **metrics,
                    }
                )
                reliability.insert(0, "split", split_name)
                reliability.insert(0, "method", method)
                reliability_outputs.append(reliability)
        except (
            ArithmeticError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            comparison_records.append(
                {
                    "method": method,
                    "split": "all",
                    "status": "failed",
                    "error": str(error),
                }
            )

    comparison = pd.DataFrame(comparison_records)
    selected_method = select_calibration_method(comparison)
    selected_calibrator = calibrators[selected_method]

    validation_output = validation_predictions.copy()
    validation_output["calibrated_probability"] = calibrated_by_method[
        (selected_method, "validation")
    ]
    test_output = test_predictions.copy()
    test_output["calibrated_probability"] = calibrated_by_method[(selected_method, "test")]
    oof_output = oof_predictions.copy()
    oof_output["calibrated_probability"] = calibrated_by_method[(selected_method, "training_oof")]

    positive_hive_count = int(
        training_rows.loc[
            training_rows[target_column].eq(1),
            HIVE_COLUMN,
        ].nunique()
    )
    gate_settings = settings["research_gate"]
    gate = evaluate_calibration_gate(
        comparison,
        selected_method=selected_method,
        minimum_validation_brier_improvement_fraction=float(
            gate_settings["minimum_validation_brier_improvement_fraction"]
        ),
        require_validation_ece_not_worse=bool(gate_settings["require_validation_ece_not_worse"]),
        maximum_test_brier_degradation_fraction=float(
            gate_settings["maximum_test_brier_degradation_fraction"]
        ),
        positive_hive_count=positive_hive_count,
        minimum_positive_hives=int(gate_settings["minimum_positive_hives"]),
    )

    comparison.to_csv(
        output_directory / "calibration_method_comparison.csv",
        index=False,
    )
    pd.concat(
        reliability_outputs,
        ignore_index=True,
    ).to_csv(
        output_directory / "calibration_reliability_bins.csv",
        index=False,
    )
    fold_audit.to_csv(
        output_directory / "grouped_oof_fold_audit.csv",
        index=False,
    )
    oof_output.to_parquet(
        output_directory / "training_oof_calibrated_predictions.parquet",
        index=False,
    )
    validation_output.to_parquet(
        output_directory / "selected_validation_calibrated_predictions.parquet",
        index=False,
    )
    test_output.to_parquet(
        output_directory / "selected_test_calibrated_predictions.parquet",
        index=False,
    )
    joblib.dump(
        selected_calibrator,
        model_directory / "selected_probability_calibrator.joblib",
    )

    indexed = comparison.set_index(["method", "split"])
    selected_summary = {
        split: {
            key: _json_safe(value)
            for key, value in indexed.loc[(selected_method, split)].to_dict().items()
            if key != "status"
        }
        for split in ("training_oof", "validation", "test")
    }
    raw_summary = {
        split: {
            key: _json_safe(value)
            for key, value in indexed.loc[("identity", split)].to_dict().items()
            if key != "status"
        }
        for split in ("training_oof", "validation", "test")
    }
    summary = {
        "research_stage": (
            "Training-only grouped out-of-fold probability calibration "
            "completed for the selected 72-hour classifier."
        ),
        "selected_classifier": selected_model_name,
        "selected_feature_set": str(metadata["feature_set"]),
        "target_column": target_column,
        "horizon_hours": horizon_hours,
        "selected_calibration_method": selected_method,
        "grouped_oof_folds_requested": requested_folds,
        "grouped_oof_folds_used": int(oof_output["calibration_fold"].nunique()),
        "positive_training_hives": positive_hive_count,
        "raw_score_metrics": raw_summary,
        "selected_calibration_metrics": selected_summary,
        "research_gate": gate,
        "limitations": [
            (
                "Grouped out-of-fold calibration keeps each hive in only "
                "one fold, but multiple hives may belong to the same "
                "apiary-level harvest session."
            ),
            (
                "Training contains only a small number of reviewed harvest "
                "events and temporal sessions."
            ),
            ("Validation contains two reviewed events and test contains one reviewed event."),
            (
                "This calibration is suitable only for a provisional "
                "academic HUI if the research gate passes."
            ),
        ],
    }
    _write_json(
        output_directory / "probability_calibration_summary.json",
        summary,
    )
    _write_json(
        output_directory / "probability_calibration_gate.json",
        gate,
    )
    _write_json(
        model_directory / "probability_calibrator_metadata.json",
        {
            "classifier_model_name": selected_model_name,
            "classifier_model_path": str(selected_model_path),
            "classifier_feature_set": str(metadata["feature_set"]),
            "classifier_horizon_hours": horizon_hours,
            "calibration_method": selected_method,
            "calibration_input": "raw_probability",
            "calibration_output": "calibrated_probability",
            "research_gate_passed": gate["gate_passed"],
            "operational_use_allowed": False,
            "next_stage": (
                "Review calibration metrics and agree the exact HUI "
                "mapping before constructing future-HUI regression targets."
            ),
        },
    )

    return {
        "status": "harvest_probability_calibration_complete",
        "selected_method": selected_method,
        "research_gate_passed": gate["gate_passed"],
        "oof_rows": len(oof_output),
        "validation_rows": len(validation_output),
        "test_rows": len(test_output),
        "comparison_path": str(output_directory / "calibration_method_comparison.csv"),
        "summary_path": str(output_directory / "probability_calibration_summary.json"),
        "gate_path": str(output_directory / "probability_calibration_gate.json"),
        "calibrator_path": str(model_directory / "selected_probability_calibrator.joblib"),
    }
