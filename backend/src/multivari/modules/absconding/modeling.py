from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .config import AbscondingSettings


class EnvironmentalStressRuleModel(BaseEstimator, ClassifierMixin):
    """Transparent report-aligned stress baseline.

    The score combines temperature instability, humidity instability, CO₂ build-up,
    long-term weight decline and the combined instability index. It is deliberately
    kept as a baseline: the selected deployment model is still learned and chosen on
    validation data.
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None):
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        def values(column: str) -> np.ndarray:
            if column not in X:
                return np.zeros(len(X), dtype=float)
            return (
                pd.to_numeric(X[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )

        temperature = np.clip(np.abs(values("temperature_c_z_72h")) / 3.0, 0.0, 1.0)
        humidity = np.clip(np.abs(values("humidity_pct_z_72h")) / 3.0, 0.0, 1.0)
        co2 = np.clip(np.maximum(values("co2_ppm_z_72h"), 0.0) / 3.0, 0.0, 1.0)
        weight = np.clip(np.maximum(-values("weight_kg_change_72h"), 0.0) / 4.0, 0.0, 1.0)
        instability = np.clip(values("multisensor_instability_index"), 0.0, 1.0)
        supplied_stress = np.clip(values("environmental_stress_score"), 0.0, 1.0)

        probability = (
            0.20 * temperature
            + 0.18 * humidity
            + 0.22 * co2
            + 0.25 * weight
            + 0.10 * instability
            + 0.05 * supplied_stress
        )
        probability = np.clip(probability, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - probability, probability])


class IsolationForestRiskModel(BaseEstimator, ClassifierMixin):
    """An unsupervised rare-pattern baseline with empirical probability-like scores."""

    def __init__(self, *, random_state: int = 42, maximum_training_rows: int = 50_000):
        self.random_state = random_state
        self.maximum_training_rows = maximum_training_rows
        self.preprocessor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        self.estimator = IsolationForest(
            n_estimators=100,
            max_samples=4096,
            contamination="auto",
            n_jobs=1,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None):
        frame = X
        if y is not None:
            normal_mask = np.asarray(y) == 0
            if normal_mask.any():
                frame = X.loc[normal_mask]
        if len(frame) > self.maximum_training_rows:
            positions = np.linspace(0, len(frame) - 1, self.maximum_training_rows, dtype=int)
            frame = frame.iloc[positions]
        transformed = self.preprocessor.fit_transform(frame)
        self.estimator.fit(transformed)
        scores = -self.estimator.score_samples(transformed)
        self.reference_scores_ = np.sort(scores)
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(X)
        scores = -self.estimator.score_samples(transformed)
        positions = np.searchsorted(self.reference_scores_, scores, side="right")
        probability = positions / max(len(self.reference_scores_), 1)
        probability = np.clip(probability, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - probability, probability])


@dataclass
class FittedCandidate:
    key: str
    display_name: str
    family: str
    estimator: Any


def build_candidate(key: str, settings: AbscondingSettings) -> FittedCandidate:
    common_imputer = SimpleImputer(strategy="median")
    if key == "dummy_prior":
        estimator = Pipeline(
            [("imputer", common_imputer), ("classifier", DummyClassifier(strategy="prior"))]
        )
        name = "Prior-probability baseline"
        family = "Baseline"
    elif key == "rule_based_stress":
        estimator = EnvironmentalStressRuleModel()
        name = "Rule-based environmental stress baseline"
        family = "Rule baseline"
    elif key == "gaussian_nb":
        estimator = Pipeline(
            [
                ("imputer", common_imputer),
                ("scale", StandardScaler()),
                ("classifier", GaussianNB(var_smoothing=1e-8)),
            ]
        )
        name = "Gaussian Naive Bayes"
        family = "Classical ML"
    elif key == "logistic_balanced":
        estimator = Pipeline(
            [
                ("imputer", common_imputer),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=1000,
                        solver="liblinear",
                        random_state=settings.random_state,
                    ),
                ),
            ]
        )
        name = "Balanced Logistic Regression"
        family = "Classical ML"
    elif key == "ridge_classifier":
        estimator = Pipeline(
            [
                ("imputer", common_imputer),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    RidgeClassifier(
                        alpha=1.0,
                        class_weight="balanced",
                        random_state=settings.random_state,
                    ),
                ),
            ]
        )
        name = "Balanced Ridge Classifier"
        family = "Classical ML"
    elif key == "decision_tree":
        estimator = Pipeline(
            [
                ("imputer", common_imputer),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=10,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=settings.random_state,
                    ),
                ),
            ]
        )
        name = "Balanced Decision Tree"
        family = "Tree model"
    elif key == "random_forest":
        estimator = Pipeline(
            [
                ("imputer", common_imputer),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=160,
                        max_depth=12,
                        min_samples_leaf=18,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=settings.random_state,
                    ),
                ),
            ]
        )
        name = "Balanced Random Forest"
        family = "Ensemble tree"
    elif key == "extra_trees":
        estimator = Pipeline(
            [
                ("imputer", common_imputer),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=120,
                        max_depth=14,
                        min_samples_leaf=12,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=settings.random_state,
                    ),
                ),
            ]
        )
        name = "Balanced Extra Trees"
        family = "Ensemble tree"
    elif key == "hist_gradient_boosting":
        estimator = Pipeline(
            [
                ("imputer", common_imputer),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=180,
                        max_leaf_nodes=24,
                        min_samples_leaf=25,
                        l2_regularization=0.5,
                        early_stopping=True,
                        random_state=settings.random_state,
                    ),
                ),
            ]
        )
        name = "Histogram Gradient Boosting"
        family = "Boosting"
    elif key == "isolation_forest":
        estimator = IsolationForestRiskModel(
            random_state=settings.random_state,
            maximum_training_rows=settings.anomaly_training_rows,
        )
        name = "Isolation Forest anomaly baseline"
        family = "Anomaly baseline"
    else:
        raise ValueError(f"Unknown absconding model candidate: {key}")
    return FittedCandidate(key=key, display_name=name, family=family, estimator=estimator)


def fit_candidate(
    candidate: FittedCandidate,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> FittedCandidate:
    if candidate.key == "hist_gradient_boosting":
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        candidate.estimator.fit(X_train, y_train, classifier__sample_weight=weights)
    else:
        candidate.estimator.fit(X_train, y_train)
    return candidate


def positive_probability(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probability = estimator.predict_proba(X)
        return np.asarray(probability[:, 1], dtype=float)
    if hasattr(estimator, "decision_function"):
        score = np.asarray(estimator.decision_function(X), dtype=float)
        score = np.clip(score, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-score))
    prediction = np.asarray(estimator.predict(X), dtype=float)
    return np.clip(prediction, 0.0, 1.0)


def stratified_training_sample(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    maximum_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep every rare positive and sample negatives reproducibly for comparison."""
    if len(X) <= maximum_rows:
        return X, y
    y_array = np.asarray(y, dtype=int)
    positive_positions = np.flatnonzero(y_array == 1)
    negative_positions = np.flatnonzero(y_array == 0)
    remaining = max(maximum_rows - len(positive_positions), 1)
    rng = np.random.default_rng(random_state)
    if len(negative_positions) > remaining:
        negative_positions = rng.choice(negative_positions, size=remaining, replace=False)
    positions = np.sort(np.concatenate([positive_positions, negative_positions]))
    return X.iloc[positions], y_array[positions]


def feature_importance(
    estimator: Any,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    feature_names: list[str],
    *,
    maximum_rows: int,
    random_state: int,
) -> list[dict[str, float | str]]:
    classifier = estimator.named_steps.get("classifier") if isinstance(estimator, Pipeline) else None
    if classifier is not None and hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
    elif classifier is not None and hasattr(classifier, "coef_"):
        values = np.abs(np.asarray(classifier.coef_[0], dtype=float))
    elif len(np.unique(y_validation)) > 1 and not isinstance(
        estimator, (IsolationForestRiskModel, EnvironmentalStressRuleModel)
    ):
        frame = X_validation
        labels = y_validation
        if len(frame) > maximum_rows:
            positions = np.linspace(0, len(frame) - 1, maximum_rows, dtype=int)
            frame = frame.iloc[positions]
            labels = labels[positions]
        result = permutation_importance(
            estimator,
            frame,
            labels,
            scoring="average_precision",
            n_repeats=3,
            random_state=random_state,
            n_jobs=1,
        )
        values = np.maximum(result.importances_mean, 0)
    else:
        values = np.zeros(len(feature_names), dtype=float)

    rows = [
        {"feature": feature, "importance": round(float(value), 8)}
        for feature, value in zip(feature_names, values, strict=True)
    ]
    return sorted(rows, key=lambda item: item["importance"], reverse=True)
