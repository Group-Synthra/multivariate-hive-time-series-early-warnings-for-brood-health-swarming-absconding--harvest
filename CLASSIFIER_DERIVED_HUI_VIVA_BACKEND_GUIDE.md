# Classifier-Derived HUI — Final Viva Backend Stage

This package replaces the earlier sensor-formula Provisional HUI experiment.

It implements the agreed final research method:

1. Use leakage-safe, Platt-adjusted classifier scores.
2. Convert those scores into a monotonic 0–100 **Classifier-Derived Provisional HUI**.
3. Create exact +24h, +48h and +72h future-HUI targets without crossing hive, split or time gaps.
4. Compare persistence, Ridge, Random Forest, XGBoost and LightGBM regressors.
5. Evaluate MAE, RMSE, median absolute error, bias, R², ±5/±10-point agreement and readiness-class agreement.

The HUI is a relative decision-support index. It is not a literal probability percentage, a verified honey-maturity label or an operationally calibrated score.

## Frozen HUI anchors

The anchors use training out-of-fold evidence only:

| Adjusted classifier score | HUI | Evidence meaning |
|---:|---:|---|
| 0 | 0 | Minimum evidence |
| 0.0008187404 | 20 | Training OOF median |
| 0.0034394735 | 40 | Exact adjusted alert boundary |
| 0.0088243793 | 60 | Positive OOF lower quartile |
| 0.0194334588 | 70 | Positive OOF median |
| 0.0509073533 | 80 | Positive OOF upper quartile |
| 0.3637653094 | 100 | Positive OOF 99th percentile |

The transformation is monotonic piecewise-linear interpolation.

## 1. Extract at the project root

From:

```powershell
C:\Users\user\Desktop\Research\multivariate-hive-time-series-early-warnings-for-brood-health-swarming-absconding--harvest
```

run:

```powershell
$package = Get-ChildItem `
  "$HOME\Downloads" `
  -Filter "classifier_derived_hui_viva_backend_package*.zip" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

Expand-Archive `
  -Path $package.FullName `
  -DestinationPath . `
  -Force
```

## 2. Merge the configuration

```powershell
cd .\backend\

python `
  scripts/merge_harvesting_classifier_derived_hui_config.py
```

Confirm:

```powershell
python -c "import yaml; c=yaml.safe_load(open('config/harvesting.yaml',encoding='utf-8')); print(c['classifier_derived_hui']['hui_anchors'])"
```

Then remove only the temporary section file:

```powershell
Remove-Item `
  .\config\harvesting_classifier_derived_hui_section.yaml `
  -ErrorAction SilentlyContinue
```

Do not rerun the merge script after deleting the temporary file.

## 3. Quality checks

```powershell
ruff check . --fix
ruff check .

pytest `
  tests/modules/harvesting/test_classifier_derived_hui.py `
  -v

pytest -v
```

The new test file contains six tests. The total project test count should increase by six relative to the current 72 tests.

## 4. Build the classifier-derived HUI dataset

```powershell
python scripts/build_classifier_derived_hui_dataset.py
```

Confirm:

```powershell
Test-Path `
  .\data\processed\classifier_derived_hui_dataset.parquet
```

Inspect the class distribution:

```powershell
python -c "import pandas as pd; p='artifacts/reports/harvesting/reviewed/classifier_derived_hui/classifier_derived_hui_distribution.csv'; print(pd.read_csv(p).to_string(index=False,float_format=lambda x:f'{x:.3f}'))"
```

Inspect future-target availability:

```powershell
python -c "import pandas as pd; p='artifacts/reports/harvesting/reviewed/classifier_derived_hui/future_hui_target_availability.csv'; print(pd.read_csv(p).to_string(index=False))"
```

Expected research behaviour:

- All HUI values remain inside 0–100.
- Validation and test may contain few or no High-Priority rows. Do not alter the frozen anchors merely to make the chart look balanced.
- Future targets must exist in train, validation and test for every horizon.

## 5. Train future-HUI regressors

```powershell
python `
  scripts/run_classifier_derived_future_hui_regression.py
```

The comparison evaluates:

- Persistence
- Ridge
- Random Forest
- XGBoost
- LightGBM

across:

- HUI history only
- Weight + HUI history
- No-humidity sensor features + HUI history

## 6. Print final evaluation values

```powershell
python `
  scripts/summarize_classifier_derived_hui_results.py
```

Save or upload the complete output. The final viva frontend should only be built after these actual values are reviewed.

## Research interpretation

A failed future-HUI gate does not mean the software failed. It means learned regressors did not beat current-HUI persistence by at least 5% on validation for two horizons while maintaining the configured test/validation consistency.

Use the actual metrics. Do not tune anchors, classes or gates only to manufacture stronger results.

## Files created by this package

```text
backend/config/harvesting_classifier_derived_hui_section.yaml
backend/scripts/merge_harvesting_classifier_derived_hui_config.py
backend/scripts/build_classifier_derived_hui_dataset.py
backend/scripts/run_classifier_derived_future_hui_regression.py
backend/scripts/summarize_classifier_derived_hui_results.py
backend/src/multivari/modules/harvesting/classifier_derived_hui.py
backend/tests/modules/harvesting/test_classifier_derived_hui.py
```
