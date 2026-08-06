"""
=========================================================
Live Prediction — LSTM Predictor
=========================================================

Responsibility:
  1. Load best_lstm.keras once (module-level singleton).
  2. Accept latest sensor readings for a hive.
  3. Build the (1, 24, 12) sequence via preprocessing.
  4. Run inference with the LSTM model (sigmoid output).
  5. Return raw probability and PELT snapshot.
  6. Risk classification is handled by RiskClassifier.

Sigmoid output note:
  The LSTM final layer is Dense(1, activation="sigmoid").
  The raw output P(swarming | x) is already a probability in [0, 1].

  risk_percentage = probability × 100
  This calculation is handled by RiskClassifier.
=========================================================
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from .config import LSTM_MODEL_PATH, OPTIMAL_THRESHOLD
from .preprocessing import build_sequence

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Module-level model singleton
# -------------------------------------------------------

_model = None


def _load_model():
    """Load the Keras model once and cache it."""
    global _model

    if _model is None:
        if not os.path.exists(LSTM_MODEL_PATH):
            raise FileNotFoundError(
                f"LSTM model not found at: {LSTM_MODEL_PATH}\nRun the LSTM training pipeline first."
            )

        # Import TensorFlow here so Flask can start even if
        # TensorFlow is unavailable during initial loading.
        from tensorflow.keras.models import load_model as keras_load

        logger.info("Loading LSTM model from %s …", LSTM_MODEL_PATH)
        _model = keras_load(LSTM_MODEL_PATH)
        logger.info("LSTM model loaded successfully.")

    return _model


# -------------------------------------------------------
# Public prediction function
# -------------------------------------------------------


def predict(hive_id: str, readings: list[dict]) -> dict:
    """
    Run live swarming prediction for a single hive.

    Parameters
    ----------
    hive_id:
        Identifier for the hive, for example, ``Hive_01``.
    readings:
        Latest sensor readings. At least 24 readings are required.

    Returns
    -------
    dict
        JSON-serializable prediction result containing the raw LSTM
        probability, predicted class, threshold, UTC timestamp, and
        latest PELT feature values.
    """
    # Step 1: Build the (1, 24, 12) input sequence.
    sequence = build_sequence(readings)

    # Step 2: Run LSTM inference.
    model = _load_model()
    raw_output = model.predict(sequence, verbose=0)
    probability = float(raw_output[0][0])

    # Step 3: Apply the trained decision threshold.
    predicted_class_int = int(probability >= OPTIMAL_THRESHOLD)
    predicted_class = "Swarming" if predicted_class_int == 1 else "No Swarming"

    # Step 4: Generate PELT features and extract the latest values.
    from .pelt_live import generate_pelt_features

    pelt_df = generate_pelt_features(readings)
    last_row = pelt_df.iloc[-1]

    pelt_snapshot = {
        "breakpoint": int(last_row["breakpoint"]),
        "days_since_breakpoint": round(
            float(last_row["days_since_breakpoint"]),
            1,
        ),
        "breakpoint_density": round(
            float(last_row["breakpoint_density"]),
            2,
        ),
        "segment_duration": round(
            float(last_row["segment_duration"]),
            1,
        ),
    }

    # Risk percentage, risk level, and warning are added later
    # by RiskClassifier.
    return {
        "hive_id": hive_id,
        "probability": round(probability, 6),
        "predicted_class": predicted_class,
        "threshold_used": OPTIMAL_THRESHOLD,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pelt_snapshot": pelt_snapshot,
    }
