# Model Weight-Scaling Fix

## Why the current dashboard result is not acceptable

All 16 candidates currently show approximately:

- PR-AUC: 0.004
- precision: 0.004
- recall: 1.000
- F1: 0.007
- validation event recall: 100%
- false-alert episodes: 50

This is effectively a no-skill, almost-all-positive operating point. The
models are not meaningfully separating event and non-event rows.

The original session-balancing code normalized all sample weights so that
their **sum was one**. That preserves relative weights but makes the total
weighted fitting loss extremely small:

- regularization dominates Logistic Regression;
- XGBoost may fail `min_child_weight` split requirements;
- LightGBM may fail weighted split requirements;
- fitted probabilities can collapse toward an intercept-only score.

The correction preserves the same relative weighting while scaling the
weights so their **mean is one**.

## Apply

Extract at the repository root, then from `backend` run:

```powershell
python scripts/apply_research_model_weight_fix.py

ruff check . --fix
ruff check .

pytest tests/modules/harvesting/test_research_model_comparison.py -v
pytest -v
```

## Remove old outputs and rerun

```powershell
Remove-Item `
  .\artifacts\reports\harvesting\reviewed\research_models `
  -Recurse `
  -Force `
  -ErrorAction SilentlyContinue

Remove-Item `
  .\artifacts\models\harvesting\research_v2 `
  -Recurse `
  -Force `
  -ErrorAction SilentlyContinue

python scripts/run_research_harvest_model_comparison.py
```

## Diagnose

```powershell
python scripts/diagnose_research_model_results.py
```

Do not proceed to probability calibration or HUI unless:

- probabilities are not effectively constant;
- at least one candidate exceeds the no-skill validation PR-AUC;
- candidate metrics are no longer identical across all models;
- event detection does not require an almost-all-positive threshold;
- grouped-hive robustness is reviewed.
