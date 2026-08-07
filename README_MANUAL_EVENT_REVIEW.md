# Manual Harvest-Event Review

The label audit found 27 delayed markers, 16 aligned markers and one event with
no clear sustained weight drop. The median delayed-marker offset was 53 hours.
The current target must therefore not be used for final model training without
review.

## Install

Copy the package's `backend` folder into the repository root.

Append `backend/config/harvesting_review_section.yaml` to the existing
`backend/config/harvesting.yaml`.

## Prepare the review sheet

From `backend`:

```powershell
ruff check . --fix
ruff check .
pytest -v
python scripts/prepare_harvest_event_review.py
```

Open:

`artifacts/reports/harvesting/label_audit/event_manual_review.csv`

Also open the 44 plots in:

`artifacts/reports/harvesting/eda/figures/individual_events/`

## Complete every row

Allowed `manual_event_type` values:

- `probable_harvest`
- `equipment_change`
- `sensor_error`
- `unclear`

For a usable harvest event:

- `manual_event_type = probable_harvest`
- `manual_include_for_training = 1`
- `manual_reviewed_event_start = exact hourly timestamp`
- `manual_review_complete = 1`

For an excluded event:

- choose the correct non-harvest type
- `manual_include_for_training = 0`
- the reviewed timestamp may be blank
- `manual_review_complete = 1`

The suggested values are aids only. Inspect each plot before copying them into
the manual columns.

## Finalize

After saving the CSV:

```powershell
python scripts/finalize_harvest_event_review.py
```

Outputs:

- `data/interim/harvest_events_reviewed.parquet`
- `artifacts/reports/harvesting/label_audit/harvest_events_reviewed.csv`
- `artifacts/reports/harvesting/label_audit/manual_review_summary.json`
