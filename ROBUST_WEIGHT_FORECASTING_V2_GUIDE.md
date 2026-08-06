# Robust Future-Weight Forecasting V2

## Why this stage is needed

The first endpoint forecast did not pass its research gate:

- the 24-hour LightGBM improvement was only about 1.8%;
- persistence remained the selected model at 48 hours;
- persistence remained the selected model at 72 hours.

The readiness builder correctly stopped. Do not lower the existing gate merely
to make the prototype run.

## Target reformulation

The original target used a single future hourly weight. A single endpoint can
contain sensor noise and short-lived disturbances.

V2 uses robust trailing-median endpoints:

```text
robust delta at horizon h
= median weight in the contiguous 6-hour window ending at t+h
- median weight in the contiguous 6-hour window ending at t
```

Both windows must:

- belong to the same hive;
- remain inside the same official split;
- contain six consecutive hourly observations.

This is still an observable future-weight target. It is not a harvest label.

## Candidate comparison

For 24, 48 and 72 hours, compare:

- persistence;
- recent-trend extrapolation;
- Ridge regression;
- Random Forest;
- XGBoost;
- LightGBM.

Feature sets:

- weight only;
- no humidity.

## Gate

Do not proceed to readiness unless:

- at least two horizons improve validation MAE over robust persistence by 2%;
- the 72-hour model is not worse than robust persistence;
- test MAE is no more than twice validation MAE.

Do not change the gate after viewing results.

## Possible outcomes

### Gate passes

Activate the robust model directory and build the already supplied provisional
readiness prototype.

### Gate fails

Stop predictive readiness development with the current dataset. Report both
classification and forecasting as benchmark experiments and present the live
interface as a monitoring/research prototype rather than a validated harvest
recommendation system.
