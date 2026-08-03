# Common dataset profile

Source workbook sheet: `Common_Dataset`

- Records: **311,044**
- Hives: **48**
- Time range: **2021-05-27 10:00:00 to 2023-02-15 23:00:00**
- Sampling interval: **hourly**
- Duplicate `(hive_id, timestamp)` pairs: **0**
- Missing values in the ten supplied columns: **0**

## Columns

| Group | Columns |
|---|---|
| Keys | `timestamp`, `hive_id` |
| Sensors | `temperature_c`, `co2_ppm`, `humidity_pct`, `weight_kg` |
| Labels | `brood_health_healthy_1`, `swarming_happened_1`, `absconding_happened_1`, `honey_harvested_1` |

## Label balance

| Label | Positive rows | Positive rate |
|---|---:|---:|
| Brood healthy | 208,471 | 67.023% |
| Swarming | 22 | 0.0071% |
| Absconding | 7 | 0.0023% |
| Honey harvested | 59 | 0.0190% |

The event labels are extremely imbalanced. Accuracy alone is not a suitable model-selection metric for swarming, absconding, or harvesting. Never apply oversampling before the chronological split.
