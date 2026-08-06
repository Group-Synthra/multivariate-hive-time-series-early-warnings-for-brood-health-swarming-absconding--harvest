# Harvest Label-Alignment Audit

The EDA shows unusually high weight variability before many generated harvest
markers. This milestone checks whether the marker timestamp is delayed relative
to the actual onset of the sustained weight drop used to create the pseudo-label.

## Install

Copy the `backend` folder into the repository root.

Append `backend/config/harvesting_label_audit_section.yaml` to the existing
`backend/config/harvesting.yaml`.

## Run

From `backend`:

```powershell
ruff check . --fix
ruff check .
pytest -v
python scripts/run_harvest_label_audit.py
```

## Outputs

- `artifacts/reports/harvesting/label_audit/event_label_alignment_audit.csv`
- `artifacts/reports/harvesting/label_audit/event_label_alignment_summary.json`

## Manual review columns

Open `event_label_alignment_audit.csv` together with the 44 event plots and fill:

- `manual_event_type`: probable_harvest / equipment_change / sensor_error / unclear
- `manual_include_for_training`: 1 or 0
- `manual_reviewed_event_start`: confirmed or corrected event timestamp
- `manual_reviewer_notes`

Do not automatically replace all marker times with detected drop onsets. The
algorithm is an audit assistant, not ground truth.
