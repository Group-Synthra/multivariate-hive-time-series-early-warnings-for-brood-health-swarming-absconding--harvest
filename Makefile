.PHONY: install test lint common-pipeline

install:
	cd backend && pip install -e ".[dev]"

test:
	cd backend && pytest

lint:
	cd backend && ruff check src tests scripts

common-pipeline:
	cd backend && python scripts/run_common_pipeline.py --input data/raw/Common_Beehive_Complete_Training_Dataset_311044.xlsx
