# """
# =========================================================
# Live Prediction — LSTM Predictor
# =========================================================

# Responsibility:
#   1. Load best_lstm.keras once (module-level singleton).
#   2. Accept latest sensor readings for a hive.
#   3. Build the (1,24,12) sequence via preprocessing.
#   4. Run inference with the LSTM model (sigmoid output).
#   5. Apply optimal threshold (0.70) to classify.
#   6. Map probability → risk percentage → risk level → message.
#   7. Return a fully structured JSON-ready dict.

# Sigmoid output note:
#   The LSTM final layer is Dense(1, activation="sigmoid").
#   The raw output P(swarming | x) is already a probability in [0,1].
 
#   risk_percentage = probability × 100
# =========================================================
# """

# import os
# import logging
# import numpy as np
# from datetime import datetime

# from .config import (
#     LSTM_MODEL_PATH,
#     OPTIMAL_THRESHOLD,
#     RISK_LOW_MAX,
#     RISK_MEDIUM_MAX,
#     RISK_MESSAGES,
# )
# from .preprocessing import build_sequence

# logger = logging.getLogger(__name__)


# # -------------------------------------------------------
# # Module-level model singleton
# # -------------------------------------------------------

# _model = None


# def _load_model():
#     """Load the Keras model once and cache it."""
#     global _model
#     if _model is None:
#         if not os.path.exists(LSTM_MODEL_PATH):
#             raise FileNotFoundError(
#                 f"LSTM model not found at: {LSTM_MODEL_PATH}\n"
#                 "Run the LSTM training pipeline first."
#             )
#         # Import TF here so Flask starts even if TF is unavailable
#         from tensorflow.keras.models import load_model as keras_load
#         logger.info("Loading LSTM model from %s …", LSTM_MODEL_PATH)
#         _model = keras_load(LSTM_MODEL_PATH)
#         logger.info("LSTM model loaded successfully.")
#     return _model


# # -------------------------------------------------------
# # Risk classification helpers
# # -------------------------------------------------------

# def _classify_risk(risk_pct: float) -> tuple:
#     """
#     Map a risk percentage to (risk_level, warning_message).

#     Thresholds:
#         0  –  30%  →  LOW
#         31 –  60%  →  MEDIUM
#         61 – 100%  →  HIGH
#     """
#     if risk_pct <= RISK_LOW_MAX:
#         level = "LOW"
#     elif risk_pct <= RISK_MEDIUM_MAX:
#         level = "MEDIUM"
#     else:
#         level = "HIGH"

#     return level, RISK_MESSAGES[level]


# # -------------------------------------------------------
# # Public prediction function
# # -------------------------------------------------------

# def predict(hive_id: str, readings: list) -> dict:
#     """
#     Run live swarming prediction for a single hive.

#     Parameters
#     ----------
#     hive_id  : str
#         Identifier for the hive (e.g. "Hive_01").
#     readings : list of dict
#         Latest sensor readings — at least 24 required.

#     Returns
#     -------
#     dict
#         JSON-serialisable prediction result:
#         {
#             "hive_id"        : str,
#             "probability"    : float,   # raw sigmoid output
#             "risk_percentage": float,   # probability × 100
#             "risk_level"     : str,     # LOW / MEDIUM / HIGH
#             "warning"        : str,
#             "predicted_class": str,     # "Swarming" | "No Swarming"
#             "threshold_used" : float,
#             "timestamp"      : str,     # ISO-8601
#             "pelt_snapshot"  : dict     # last PELT feature values
#         }
#     """
#     # ── Step 1: Build (1, 24, 12) sequence ──────────────────────
#     sequence = build_sequence(readings)          # (1, 24, 12)

#     # ── Step 2: Run LSTM inference ───────────────────────────────
#     model = _load_model()
#     raw_output = model.predict(sequence, verbose=0)  # shape (1, 1)
#     probability = float(raw_output[0][0])

#     # ── Step 3: Apply threshold ──────────────────────────────────
#     predicted_class_int = int(probability >= OPTIMAL_THRESHOLD)
#     predicted_class = "Swarming" if predicted_class_int == 1 else "No Swarming"

#     # ── Step 4: Derive risk metrics ───────────────────────────────
#     risk_pct         = round(probability * 100, 2)
#     risk_level, warning = _classify_risk(risk_pct)

#     # ── Step 5: Extract PELT snapshot from last reading ───────────
#     from .pelt_live import generate_pelt_features
#     pelt_df = generate_pelt_features(readings)
#     last_row = pelt_df.iloc[-1]
#     pelt_snapshot = {
#         "breakpoint"          : int(last_row["breakpoint"]),
#         "days_since_breakpoint": round(float(last_row["days_since_breakpoint"]), 1),
#         "breakpoint_density"  : round(float(last_row["breakpoint_density"]), 2),
#         "segment_duration"    : round(float(last_row["segment_duration"]), 1),
#     }

#     return {
#         "hive_id"        : hive_id,
#         "probability"    : round(probability, 6),
#         "risk_percentage": risk_pct,
#         "risk_level"     : risk_level,
#         "warning"        : warning,
#         "predicted_class": predicted_class,
#         "threshold_used" : OPTIMAL_THRESHOLD,
#         "timestamp"      : datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
#         "pelt_snapshot"  : pelt_snapshot,
#     }

"""
=========================================================
Live Prediction — LSTM Predictor
=========================================================

Responsibility:
  1. Load best_lstm.keras once (module-level singleton).
  2. Accept latest sensor readings for a hive.
  3. Build the (1,24,12) sequence via preprocessing.
  4. Run inference with the LSTM model (sigmoid output).
  5. Return raw probability and PELT snapshot.
  6. Risk classification is now handled by RiskClassifier.

Sigmoid output note:
  The LSTM final layer is Dense(1, activation="sigmoid").
  The raw output P(swarming | x) is already a probability in [0,1].
 
  risk_percentage = probability × 100 (now calculated by RiskClassifier)
=========================================================
"""

import os
import logging
import numpy as np
from datetime import datetime

from .config import (
    LSTM_MODEL_PATH,
    OPTIMAL_THRESHOLD,
)
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
                f"LSTM model not found at: {LSTM_MODEL_PATH}\n"
                "Run the LSTM training pipeline first."
            )
        # Import TF here so Flask starts even if TF is unavailable
        from tensorflow.keras.models import load_model as keras_load
        logger.info("Loading LSTM model from %s …", LSTM_MODEL_PATH)
        _model = keras_load(LSTM_MODEL_PATH)
        logger.info("LSTM model loaded successfully.")
    return _model


# -------------------------------------------------------
# REMOVED: _classify_risk() - Now handled by RiskClassifier
# -------------------------------------------------------
# The static threshold logic has been removed. 
# Risk classification is now done by RiskClassifier 
# using Softmax probability distribution.


# -------------------------------------------------------
# Public prediction function
# -------------------------------------------------------

def predict(hive_id: str, readings: list) -> dict:
    """
    Run live swarming prediction for a single hive.

    Parameters
    ----------
    hive_id  : str
        Identifier for the hive (e.g. "Hive_01").
    readings : list of dict
        Latest sensor readings — at least 24 required.

    Returns
    -------
    dict
        JSON-serialisable prediction result:
        {
            "hive_id"        : str,
            "probability"    : float,   # raw sigmoid output
            "predicted_class": str,     # "Swarming" | "No Swarming"
            "threshold_used" : float,
            "timestamp"      : str,     # ISO-8601
            "pelt_snapshot"  : dict,    # last PELT feature values
            # REMOVED: risk_percentage, risk_level, warning
            # These are now added by RiskClassifier in swarming_api.py
        }
    """
    # ── Step 1: Build (1, 24, 12) sequence ──────────────────────
    sequence = build_sequence(readings)          # (1, 24, 12)

    # ── Step 2: Run LSTM inference ───────────────────────────────
    model = _load_model()
    raw_output = model.predict(sequence, verbose=0)  # shape (1, 1)
    probability = float(raw_output[0][0])

    # ── Step 3: Apply threshold for predicted class ──────────────
    predicted_class_int = int(probability >= OPTIMAL_THRESHOLD)
    predicted_class = "Swarming" if predicted_class_int == 1 else "No Swarming"

    # ── Step 4: REMOVED: Risk metrics calculation ────────────────
    # risk_pct = round(probability * 100, 2)
    # risk_level, warning = _classify_risk(risk_pct)  # ← REMOVED
    # These are now handled by RiskClassifier

    # ── Step 5: Extract PELT snapshot from last reading ───────────
    from .pelt_live import generate_pelt_features
    pelt_df = generate_pelt_features(readings)
    last_row = pelt_df.iloc[-1]
    pelt_snapshot = {
        "breakpoint"          : int(last_row["breakpoint"]),
        "days_since_breakpoint": round(float(last_row["days_since_breakpoint"]), 1),
        "breakpoint_density"  : round(float(last_row["breakpoint_density"]), 2),
        "segment_duration"    : round(float(last_row["segment_duration"]), 1),
    }

    return {
        "hive_id"        : hive_id,
        "probability"    : round(probability, 6),
        "predicted_class": predicted_class,
        "threshold_used" : OPTIMAL_THRESHOLD,
        "timestamp"      : datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pelt_snapshot"  : pelt_snapshot,
        # REMOVED: risk_percentage, risk_level, warning
        # These are now added by RiskClassifier in swarming_api.py
    }