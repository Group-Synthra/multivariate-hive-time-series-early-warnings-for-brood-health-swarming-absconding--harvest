# live_prediction package
"""
=========================================================
Live Prediction Package
=========================================================

Exports:
    - predict: LSTM prediction function
    - build_sequence: Build LSTM input sequences
    - generate_pelt_features: Generate PELT features
    - RiskClassifier: Risk classification with Softmax
    - create_risk_classifier: Factory function for RiskClassifier
=========================================================
"""

# ── Core prediction functions ──────────────────────────────
# ── Configuration ────────────────────────────────────────────
from .config import (
    FEATURE_COLUMNS,
    LABEL_ENCODER_PATH,
    LSTM_MODEL_PATH,
    OPTIMAL_THRESHOLD,
    RISK_LOW_MAX,
    RISK_MEDIUM_MAX,
    RISK_MESSAGES,
    SCALER_PATH,
    SEQUENCE_LENGTH,
)
from .live_predictor import predict
from .pelt_live import generate_pelt_features
from .preprocessing import build_sequence

# ── Risk Classifier with Softmax ────────────────────────────
from .risk_classifier import RiskClassifier, create_risk_classifier

# ── What gets exported with "from live_prediction import *" ──
__all__ = [
    "FEATURE_COLUMNS",
    "LABEL_ENCODER_PATH",
    # Configuration
    "LSTM_MODEL_PATH",
    "OPTIMAL_THRESHOLD",
    "RISK_LOW_MAX",
    "RISK_MEDIUM_MAX",
    "RISK_MESSAGES",
    "SCALER_PATH",
    "SEQUENCE_LENGTH",
    # Risk Classifier
    "RiskClassifier",
    "build_sequence",
    "create_risk_classifier",
    "generate_pelt_features",
    # Core functions
    "predict",
]

# ── Package metadata ─────────────────────────────────────────
__version__ = "1.0.0"
__author__ = "HiveEDA Team"
__description__ = "Live prediction module for honey bee swarming detection"

# ── Optional: Log package initialization ────────────────────
import logging

logger = logging.getLogger(__name__)
logger.debug(f"Live Prediction package v{__version__} initialized")
