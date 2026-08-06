"""
=========================================================
Honey Bee Swarming Prediction
Configuration File
=========================================================
"""

import os
from pathlib import Path

# ----------------------------------------------------
# Base Paths
# ----------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[5]
DATA_FOLDER = str(BACKEND_DIR / "data" / "swarming")
OUTPUT_FOLDER = str(BACKEND_DIR / "artifacts" / "metrics" / "swarming")
MODEL_FOLDER = str(BACKEND_DIR / "artifacts" / "models" / "swarming")
GRAPH_FOLDER = str(BACKEND_DIR / "artifacts" / "reports" / "swarming" / "graphs")

# Create folders automatically
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# ----------------------------------------------------
# Dataset
# ----------------------------------------------------

DATASET_PATH = os.path.join(
    DATA_FOLDER,
    "hive_data_with_features.csv"
)

# ----------------------------------------------------
# Target Column
# ----------------------------------------------------

TARGET_COLUMN = "swarming_label_next_72h"

# ----------------------------------------------------
# Hive Column
# ----------------------------------------------------

HIVE_COLUMN = "hive_id"

# ----------------------------------------------------
# Timestamp Column
# ----------------------------------------------------

TIMESTAMP_COLUMN = "timestamp"

# ----------------------------------------------------
# Sensor Features
# ----------------------------------------------------

FEATURE_COLUMNS = [

    "internal_temperature_c",

    "internal_humidity_pct",

    "co2_ppm",

    "hive_weight_kg",

    "external_temperature_c",

    "external_humidity_pct",

    "rainfall_mm_hour",

    "wind_speed_mps"

]

# ----------------------------------------------------
# PELT Features
# ----------------------------------------------------

PELT_COLUMNS = [

    "internal_temperature_c",

    "internal_humidity_pct",

    "co2_ppm",

    "hive_weight_kg"

]

# ----------------------------------------------------
# Random State
# ----------------------------------------------------

RANDOM_STATE = 42

# ----------------------------------------------------
# Train/Test Split
# ----------------------------------------------------

TRAIN_RATIO = 0.80

# ----------------------------------------------------
# LSTM
# ----------------------------------------------------

SEQUENCE_LENGTH = 24

BATCH_SIZE = 64

EPOCHS = 100

PATIENCE = 10
