"""
=========================================================
Live Prediction — Preprocessing
=========================================================

Responsibility:
  1. Load the saved StandardScaler (lstm_scaler.pkl) once.
  2. Load the saved LabelEncoder (label_encoder.pkl) once.
  3. Expose build_sequence() which:
       a. Calls pelt_live.generate_pelt_features()
       b. Scales the 12-feature DataFrame with the saved scaler
       c. Returns a numpy array of shape (1, 24, 12) ready
          for LSTM inference.
=========================================================
"""

import numpy as np
import joblib

from .config import (
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
)
from .pelt_live import generate_pelt_features


# -------------------------------------------------------
# Module-level singletons — loaded once at import time
# -------------------------------------------------------

def _load_scaler():
    """Load and return the saved StandardScaler."""
    try:
        return joblib.load(SCALER_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"StandardScaler not found at: {SCALER_PATH}\n"
            "Run the LSTM training pipeline first."
        )


def _load_label_encoder():
    """Load and return the saved LabelEncoder."""
    try:
        return joblib.load(LABEL_ENCODER_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"LabelEncoder not found at: {LABEL_ENCODER_PATH}\n"
            "Run the LSTM training pipeline first."
        )


# Lazy-load on first call to avoid errors if files do not exist yet
_scaler        = None
_label_encoder = None


def get_scaler():
    """Return the module-level scaler, loading it on first call."""
    global _scaler
    if _scaler is None:
        _scaler = _load_scaler()
    return _scaler


def get_label_encoder():
    """Return the module-level label encoder, loading it on first call."""
    global _label_encoder
    if _label_encoder is None:
        _label_encoder = _load_label_encoder()
    return _label_encoder


# -------------------------------------------------------
# Public API
# -------------------------------------------------------

def build_sequence(readings: list) -> np.ndarray:
    """
    Convert raw sensor readings into a scaled LSTM input sequence.

    Parameters
    ----------
    readings : list of dict
        Latest sensor readings (at least 24 entries required).
        Each dict must contain the 8 sensor keys.

    Returns
    -------
    np.ndarray
        Shape: (1, 24, 12)
        Scaled feature sequence ready for model.predict().
    """
    # Step 1: Generate 12-column feature DataFrame via PELT
    feature_df = generate_pelt_features(readings)

    # Verify shape
    assert feature_df.shape == (SEQUENCE_LENGTH, len(FEATURE_COLUMNS)), (
        f"Expected ({SEQUENCE_LENGTH}, {len(FEATURE_COLUMNS)}), "
        f"got {feature_df.shape}"
    )

    # Step 2: Scale with the saved StandardScaler
    # Pass the DataFrame (not .values) so sklearn sees the column names it was
    # fitted with during training — avoids a spurious UserWarning.
    scaler = get_scaler()
    scaled = scaler.transform(feature_df)

    # Step 3: Reshape to (1, 24, 12) for LSTM
    sequence = scaled.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))

    return sequence.astype(np.float32)


def decode_label(encoded_label: int) -> str:
    """Decode an integer label back to its string class name."""
    encoder = get_label_encoder()
    return encoder.inverse_transform([encoded_label])[0]
