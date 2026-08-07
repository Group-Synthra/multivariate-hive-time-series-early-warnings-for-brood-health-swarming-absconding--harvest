# Classification Generalization Review and Finalization

## Evidence now available

The validation-selected alert policy:

- detected 2 of 2 reviewed validation events;
- used a 12-hour smoothing window;
- required 4 consecutive threshold hours;
- used threshold 0.0094529143922651.

Applied unchanged to the held-out test event, it detected 0 of 1 events.

This means the validation policy was eligible inside the validation gate, but
the available held-out case does not support calibration or deployment.

## Run order

From `backend`:

```powershell
ruff check . --fix
ruff check .

pytest tests/modules/harvesting/test_harvest_policy_generalization.py -v
pytest -v

python scripts/review_harvest_policy_generalization.py

python scripts/finalize_harvesting_evidence_status.py

python scripts/create_prospective_harvest_validation_template.py

python scripts/export_harvesting_benchmark_dashboard.py
```

## Expected classification review

```text
status: validation_eligible_test_event_missed
validation_gate_passed: true
unchanged_test_event_supported: false
generalization_supported: false
calibration_allowed: false
deployment_allowed: false
```

## Final research decision

The final status is benchmark-only because:

- the validation-selected classifier policy missed the unchanged test event;
- robust weight forecasting improved only the 24-hour horizon;
- persistence remained best at 48 and 72 hours.

The 24-hour weight forecast may remain visible as an exploratory historical
test-split case study.

## Dashboard wording

Approved:

- Validation alert policy eligible
- Held-out test event missed
- Calibration not supported
- Exploratory 24-hour hive-weight forecast
- Prospective validation required

Blocked:

- Calibrated harvest probability
- HUI
- Validated readiness
- Ready to harvest
- Recommended harvest time
- Live harvest recommendation
