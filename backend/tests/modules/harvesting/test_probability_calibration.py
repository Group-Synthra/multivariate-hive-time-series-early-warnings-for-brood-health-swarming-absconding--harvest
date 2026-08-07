from __future__ import annotations

import numpy as np
import pandas as pd

from multivari.modules.harvesting.probability_calibration import (
    assign_grouped_hive_folds,
    build_reliability_table,
    calculate_calibration_metrics,
    evaluate_calibration_gate,
    fit_calibrator,
    select_calibration_method,
)


def test_grouped_folds_keep_hives_together_and_balance_positives() -> None:
    rows = pd.DataFrame(
        {
            "hive_id": (
                ["hive_a"] * 8
                + ["hive_b"] * 8
                + ["hive_c"] * 8
                + ["hive_d"] * 8
            ),
            "target": (
                [0, 0, 0, 0, 1, 1, 1, 1]
                + [0, 0, 0, 0, 0, 1, 1, 1]
                + [0] * 8
                + [0] * 8
            ),
        }
    )

    fold_ids, audit = assign_grouped_hive_folds(
        rows,
        target_column="target",
        requested_folds=2,
    )

    assigned = rows.assign(fold=fold_ids)
    assert assigned.groupby("hive_id")["fold"].nunique().max() == 1
    positive_hives = audit.loc[
        audit["positive_rows"].gt(0)
    ].groupby("fold").size()
    assert positive_hives.min() >= 1


def test_calibrators_return_bounded_probabilities() -> None:
    raw = np.array(
        [0.001, 0.01, 0.05, 0.2, 0.4, 0.7],
        dtype=float,
    )
    target = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    for method in ("identity", "platt", "isotonic"):
        calibrator = fit_calibrator(
            method,
            raw,
            target,
            epsilon=1e-6,
        )
        calibrated = calibrator.predict(raw)
        assert calibrated.shape == raw.shape
        assert np.isfinite(calibrated).all()
        assert (calibrated >= 0.0).all()
        assert (calibrated <= 1.0).all()


def test_reliability_table_accounts_for_every_row() -> None:
    target = np.array([0, 0, 1, 0, 1, 1], dtype=int)
    probabilities = np.array(
        [0.05, 0.10, 0.25, 0.40, 0.70, 0.90],
        dtype=float,
    )

    table = build_reliability_table(
        target,
        probabilities,
        requested_bins=3,
    )

    assert int(table["rows"].sum()) == len(target)
    assert int(table["positive_rows"].sum()) == int(target.sum())


def test_calibration_metrics_report_probability_quality() -> None:
    target = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    probabilities = np.array(
        [0.05, 0.10, 0.20, 0.70, 0.80, 0.90],
        dtype=float,
    )

    metrics, reliability = calculate_calibration_metrics(
        target,
        probabilities,
        reliability_bins=3,
        epsilon=1e-6,
    )

    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    assert 0.0 <= metrics["expected_calibration_error"] <= 1.0
    assert not reliability.empty


def test_selection_uses_validation_metrics_not_test_metrics() -> None:
    comparison = pd.DataFrame(
        [
            {
                "method": "identity",
                "split": "validation",
                "status": "ok",
                "brier_score": 0.20,
                "log_loss": 0.50,
                "expected_calibration_error": 0.10,
            },
            {
                "method": "platt",
                "split": "validation",
                "status": "ok",
                "brier_score": 0.10,
                "log_loss": 0.30,
                "expected_calibration_error": 0.05,
            },
            {
                "method": "isotonic",
                "split": "validation",
                "status": "ok",
                "brier_score": 0.15,
                "log_loss": 0.35,
                "expected_calibration_error": 0.06,
            },
            {
                "method": "identity",
                "split": "test",
                "status": "ok",
                "brier_score": 0.01,
                "log_loss": 0.05,
                "expected_calibration_error": 0.01,
            },
            {
                "method": "platt",
                "split": "test",
                "status": "ok",
                "brier_score": 0.50,
                "log_loss": 1.20,
                "expected_calibration_error": 0.40,
            },
            {
                "method": "isotonic",
                "split": "test",
                "status": "ok",
                "brier_score": 0.02,
                "log_loss": 0.06,
                "expected_calibration_error": 0.02,
            },
        ]
    )

    assert select_calibration_method(comparison) == "platt"


def test_gate_requires_real_validation_improvement() -> None:
    comparison = pd.DataFrame(
        [
            {
                "method": "identity",
                "split": "validation",
                "brier_score": 0.10,
                "expected_calibration_error": 0.05,
            },
            {
                "method": "platt",
                "split": "validation",
                "brier_score": 0.08,
                "expected_calibration_error": 0.04,
            },
            {
                "method": "identity",
                "split": "test",
                "brier_score": 0.09,
                "expected_calibration_error": 0.05,
            },
            {
                "method": "platt",
                "split": "test",
                "brier_score": 0.095,
                "expected_calibration_error": 0.05,
            },
        ]
    )

    gate = evaluate_calibration_gate(
        comparison,
        selected_method="platt",
        minimum_validation_brier_improvement_fraction=0.02,
        require_validation_ece_not_worse=True,
        maximum_test_brier_degradation_fraction=0.10,
        positive_hive_count=8,
        minimum_positive_hives=4,
    )

    assert gate["gate_passed"] is True
    assert gate["operational_calibration_allowed"] is False
