# """
# =========================================================
# Live Prediction — Configuration
# =========================================================
# Central configuration for paths, PELT parameters, and
# risk thresholds. Matches training pipeline exactly.
# =========================================================
# """

# import os

# # -------------------------------------------------------
# # Base Paths
# # -------------------------------------------------------

# # backend/live_prediction/config.py → go up two levels → backend/
# _BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MODEL_DIR = os.path.join(
#     _BACKEND_DIR,
#     "outputs",
#     "model_training",
#     "models"
# )

# # -------------------------------------------------------
# # Saved Model Artifacts
# # -------------------------------------------------------

# LSTM_MODEL_PATH   = os.path.join(MODEL_DIR, "best_lstm.keras")
# SCALER_PATH       = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
# LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# # -------------------------------------------------------
# # PELT Parameters  (must match training exactly)
# # -------------------------------------------------------

# PELT_MODEL   = "l2"   # ruptures cost model
# PELT_PEN     = 10     # penalty value used in predict()

# # Columns used to build the multivariate PELT signal
# PELT_COLUMNS = [
#     "internal_temperature_c",
#     "internal_humidity_pct",
#     "co2_ppm",
#     "hive_weight_kg",
# ]

# # -------------------------------------------------------
# # LSTM Sequence
# # -------------------------------------------------------

# SEQUENCE_LENGTH = 24   # sliding window of 24 readings

# # Optimal decision threshold found during training
# OPTIMAL_THRESHOLD = 0.70

# # -------------------------------------------------------
# # All 12 Features  (sensor + PELT) — order must match scaler
# # -------------------------------------------------------

# FEATURE_COLUMNS = [
#     "internal_temperature_c",
#     "internal_humidity_pct",
#     "co2_ppm",
#     "hive_weight_kg",
#     "external_temperature_c",
#     "external_humidity_pct",
#     "rainfall_mm_hour",
#     "wind_speed_mps",
#     "breakpoint",
#     "days_since_breakpoint",
#     "breakpoint_density",
#     "segment_duration",
# ]

# # -------------------------------------------------------
# # Risk Level Thresholds (percentage 0–100)
# # -------------------------------------------------------

# RISK_LOW_MAX    = 30
# RISK_MEDIUM_MAX = 60
# # above 60% → HIGH

# RISK_MESSAGES = {
#     "LOW":    "Hive behaviour is normal. No immediate swarming risk detected.",
#     "MEDIUM": "Possible behavioural changes detected. Continue monitoring hive.",
#     "HIGH":   "High swarming probability detected. Immediate hive inspection recommended.",
# }

"""
=========================================================
Live Prediction — Configuration
=========================================================
Central configuration for paths, PELT parameters, and
risk thresholds. Matches training pipeline exactly.
=========================================================
"""

import os
from pathlib import Path

# -------------------------------------------------------
# Base Paths
# -------------------------------------------------------

_BACKEND_DIR = Path(__file__).resolve().parents[5]
MODEL_DIR = str(_BACKEND_DIR / "artifacts" / "models" / "swarming")

# -------------------------------------------------------
# Saved Model Artifacts
# -------------------------------------------------------

LSTM_MODEL_PATH   = os.path.join(MODEL_DIR, "best_lstm.keras")
SCALER_PATH       = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# -------------------------------------------------------
# PELT Parameters  (must match training exactly)
# -------------------------------------------------------

PELT_MODEL   = "l2"   # ruptures cost model
PELT_PEN     = 10     # penalty value used in predict()

# Columns used to build the multivariate PELT signal
PELT_COLUMNS = [
    "internal_temperature_c",
    "internal_humidity_pct",
    "co2_ppm",
    "hive_weight_kg",
]

# -------------------------------------------------------
# LSTM Sequence
# -------------------------------------------------------

SEQUENCE_LENGTH = 24   # sliding window of 24 readings

# Optimal decision threshold found during training
OPTIMAL_THRESHOLD = 0.70

# -------------------------------------------------------
# All 12 Features  (sensor + PELT) — order must match scaler
# -------------------------------------------------------

FEATURE_COLUMNS = [
    "internal_temperature_c",
    "internal_humidity_pct",
    "co2_ppm",
    "hive_weight_kg",
    "external_temperature_c",
    "external_humidity_pct",
    "rainfall_mm_hour",
    "wind_speed_mps",
    "breakpoint",
    "days_since_breakpoint",
    "breakpoint_density",
    "segment_duration",
]

# -------------------------------------------------------
# Risk Level Thresholds (percentage 0–100)
# NOW USED BY: RiskClassifier (risk_classifier.py)
# -------------------------------------------------------
# These thresholds are used by the RiskClassifier to map
# risk percentages to LOW, MEDIUM, HIGH risk levels.
# 
# The RiskClassifier uses Softmax probability:
# P(swarming | x) = e^(z_swarm) / (e^(z_swarm) + e^(z_non-swarm))
#
# Risk Percentage = P(swarming | x) × 100
#
# Thresholds:
#   0%  –  30%  →  LOW RISK
#   31% –  60%  →  MEDIUM RISK
#   61% – 100%  →  HIGH RISK
# -------------------------------------------------------

RISK_LOW_MAX    = 30
RISK_MEDIUM_MAX = 60
# above 60% → HIGH

# Risk messages for each level
RISK_MESSAGES = {
    "LOW":    "Hive behaviour is normal. No immediate swarming risk detected.",
    "MEDIUM": "Possible behavioural changes detected. Continue monitoring hive.",
    "HIGH":   "High swarming probability detected. Immediate hive inspection recommended.",
}
