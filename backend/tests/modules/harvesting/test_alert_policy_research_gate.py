import pandas as pd

from multivari.modules.harvesting.alert_policy_research_gate import (
    add_operational_metrics,
    select_research_safe_policy,
)


def test_operational_metrics_identify_always_on_policy() -> None:
    sweep = pd.DataFrame(
        {
            "precision": [0.004],
            "false_positives": [38841],
            "true_negatives": [700],
            "alert_rows": [38985],
            "true_positives": [144],
            "false_negatives": [0],
        }
    )

    result = add_operational_metrics(
        sweep,
        prevalence=144 / 39685,
    )

    assert result.loc[0, "alert_fraction"] > 0.98
    assert result.loc[0, "false_positive_rate"] > 0.98


def test_gate_rejects_trivial_lower_precision_policy() -> None:
    sweep = pd.DataFrame(
        {
            "smoothing_window_hours": [12],
            "minimum_consecutive_hours": [4],
            "threshold": [0.000038],
            "precision": [0.00369],
            "recall": [1.0],
            "f1": [0.00736],
            "true_positives": [144],
            "false_positives": [38841],
            "false_negatives": [0],
            "true_negatives": [700],
            "alert_rows": [38985],
            "event_recall": [1.0],
            "median_lead_hours": [72.0],
            "false_alert_episodes": [50],
        }
    )
    baseline = {
        "precision": 0.01583,
        "true_positives": 63,
        "false_positives": 3916,
        "false_negatives": 81,
        "true_negatives": 35625,
        "alert_rows": 3979,
    }

    selected, _, gate = select_research_safe_policy(
        sweep,
        prevalence=144 / 39685,
        baseline_metrics=baseline,
        minimum_event_recall=1.0,
        minimum_median_lead_hours=12,
        minimum_precision_lift=2.0,
        require_precision_at_least_baseline=True,
        require_false_positive_rows_no_worse_than_baseline=True,
        require_alert_fraction_no_worse_than_baseline=True,
    )

    assert selected is None
    assert gate["eligible_candidate_count"] == 0


def test_gate_accepts_policy_that_improves_baseline() -> None:
    sweep = pd.DataFrame(
        {
            "smoothing_window_hours": [6],
            "minimum_consecutive_hours": [3],
            "threshold": [0.02],
            "precision": [0.03],
            "recall": [0.25],
            "f1": [0.0536],
            "true_positives": [36],
            "false_positives": [1164],
            "false_negatives": [108],
            "true_negatives": [38377],
            "alert_rows": [1200],
            "event_recall": [1.0],
            "median_lead_hours": [24.0],
            "false_alert_episodes": [40],
        }
    )
    baseline = {
        "precision": 0.01583,
        "true_positives": 63,
        "false_positives": 3916,
        "false_negatives": 81,
        "true_negatives": 35625,
        "alert_rows": 3979,
    }

    selected, _, gate = select_research_safe_policy(
        sweep,
        prevalence=144 / 39685,
        baseline_metrics=baseline,
        minimum_event_recall=1.0,
        minimum_median_lead_hours=12,
        minimum_precision_lift=2.0,
        require_precision_at_least_baseline=True,
        require_false_positive_rows_no_worse_than_baseline=True,
        require_alert_fraction_no_worse_than_baseline=True,
    )

    assert selected is not None
    assert gate["eligible_candidate_count"] == 1
