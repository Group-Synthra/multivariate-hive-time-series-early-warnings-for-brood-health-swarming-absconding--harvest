# Absconding Module V2 — Implementation Guide

The current version uses the shared cleaned dataset and chronological split manifest, a leakage-safe **next-24-hour** Absconding target, 168 hours of prior sensor history, report-aligned classical model comparisons, ARM, explainable stress factors, and a real Supabase PostgreSQL IoT inference path.

For the complete setup, environment variables, database mapping, API endpoints and exact file locations, read:

- [`ABSCONDING_IOT_UI_UPDATE_GUIDE.md`](ABSCONDING_IOT_UI_UPDATE_GUIDE.md)

Core commands from `backend`:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts/run_common_pipeline.py --input "data/raw/Common_Beehive_Complete_Training_Dataset_311044.xlsx"
python scripts/run_absconding_pipeline.py
python -m pytest
python -m ruff check src tests scripts app.py
python app.py
```

Frontend commands:

```powershell
cd frontend
npm.cmd install
npm.cmd run build
npm.cmd run dev
```

The historical data contains very few confirmed Absconding episodes. Report all results as exploratory until more correctly labelled local events are available.
