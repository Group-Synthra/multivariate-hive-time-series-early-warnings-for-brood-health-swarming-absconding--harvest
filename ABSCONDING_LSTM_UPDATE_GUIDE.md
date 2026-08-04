# Absconding LSTM Update

Apply this patch after the Absconding IoT/UI patch.

## Updated files

- `backend/pyproject.toml`
- `backend/config/absconding.yaml`
- `backend/src/multivari/modules/absconding/pipeline.py`

## New files

- `backend/src/multivari/modules/absconding/lstm.py`
- `backend/scripts/run_absconding_lstm.py`
- `backend/tests/test_absconding_lstm.py`

## Install

From `backend` with `.venv` activated:

```powershell
python -m pip install -e ".[dev,lstm]"
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Run

The classical pipeline must be run first:

```powershell
python scripts/run_absconding_pipeline.py
```

Then train the LSTM:

```powershell
python scripts/run_absconding_lstm.py --epochs 30 --sequence-length 72 --stride 3
```

Run the classical pipeline once more so its generated plots and JSON are refreshed with the saved LSTM comparison:

```powershell
python scripts/run_absconding_pipeline.py
```

## Outputs

- `artifacts/models/absconding/absconding_lstm_sequence.keras`
- `artifacts/models/absconding/absconding_lstm_preprocessor.joblib`
- `artifacts/metrics/absconding/lstm_comparison.json`
- `artifacts/metrics/absconding/lstm_training_history.csv`
- `artifacts/metrics/absconding/lstm_validation_event_detection.csv`
- `artifacts/metrics/absconding/lstm_test_event_detection.csv`

The LSTM is included in the model-comparison interface. The live IoT endpoint remains on the selected tabular deployment model until the LSTM is explicitly promoted after defensible evaluation.
