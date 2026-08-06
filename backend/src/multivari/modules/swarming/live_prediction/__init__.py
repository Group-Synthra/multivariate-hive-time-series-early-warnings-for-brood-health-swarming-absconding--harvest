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
from .live_predictor import predict
from .preprocessing import build_sequence
from .pelt_live import generate_pelt_features

# ── Risk Classifier with Softmax ────────────────────────────
from .risk_classifier import RiskClassifier, create_risk_classifier

# ── Configuration ────────────────────────────────────────────
from .config import (
    LSTM_MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    OPTIMAL_THRESHOLD,
    RISK_LOW_MAX,
    RISK_MEDIUM_MAX,
    RISK_MESSAGES,
)

# ── What gets exported with "from live_prediction import *" ──
__all__ = [
    # Core functions
    'predict',
    'build_sequence',
    'generate_pelt_features',
    
    # Risk Classifier
    'RiskClassifier',
    'create_risk_classifier',
    
    # Configuration
    'LSTM_MODEL_PATH',
    'SCALER_PATH',
    'LABEL_ENCODER_PATH',
    'FEATURE_COLUMNS',
    'SEQUENCE_LENGTH',
    'OPTIMAL_THRESHOLD',
    'RISK_LOW_MAX',
    'RISK_MEDIUM_MAX',
    'RISK_MESSAGES',
]

# ── Package metadata ─────────────────────────────────────────
__version__ = "1.0.0"
__author__ = "HiveEDA Team"
__description__ = "Live prediction module for honey bee swarming detection"

# ── Optional: Log package initialization ────────────────────
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Live Prediction package v{__version__} initialized")