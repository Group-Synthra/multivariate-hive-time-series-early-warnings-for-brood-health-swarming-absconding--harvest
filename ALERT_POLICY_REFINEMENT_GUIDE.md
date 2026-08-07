# Harvest Alert-Policy Refinement

This milestone converts the selected model's hourly raw scores into a more
operational temporal alert policy. It does not retrain the classifier and does
not calibrate HUI.

## Evaluated policy grid

- trailing score smoothing: 1, 3, 6 and 12 hours;
- required consecutive threshold hours: 1, 2, 3 and 4;
- data-derived score thresholds.

A policy is eligible when it detects both validation events and provides at
least 12 hours median validation lead time. Among eligible policies, the
selection order is:

1. fewer false-alert episodes;
2. higher precision;
3. higher F1;
4. longer lead time;
5. simpler smoothing and persistence settings.

The selected policy is applied unchanged to the one-event test split.

## Outputs

```text
artifacts/reports/harvesting/reviewed/alert_policy/
├── alert_policy_sweep.csv
├── top_alert_policies.csv
├── selected_alert_policy.json
├── selected_validation_alert_predictions.parquet
├── selected_test_alert_predictions.parquet
├── selected_validation_event_detection.csv
└── selected_test_event_detection.csv
```

Deployment metadata:

```text
artifacts/models/harvesting/research_v2/alert_policy.json
```

The script marks `ready_for_calibration_review` true only when the validation
constraints are met, false-alert episodes fall by at least 50%, and alert
precision is at least twice the validation prevalence. This is a project gate,
not proof of external validity.
