"""
=========================================================
Honey Bee Swarming Prediction
Preprocessing Module
=========================================================


Project : Honey Bee Swarming Prediction Dashboard

Purpose:
1. Load dataset
2. Clean data
3. Select features
4. Encode labels
5. Scale features
6. Split train/test
7. Save processed data

=========================================================
"""

import os

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import DATASET_PATH, OUTPUT_FOLDER

print("=" * 60)
print("Honey Bee Swarming Prediction")
print("Data Preprocessing Module")
print("=" * 60)

# -------------------------------------------------------
# Create Output Folder
# -------------------------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Output folder ready.")

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")

print("\nRows :", df.shape[0])
print("Columns :", df.shape[1])

# -------------------------------------------------------
# Convert Timestamp
# -------------------------------------------------------

print("\nConverting timestamp...")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp")

df = df.reset_index(drop=True)

print("Timestamp converted.")

# -------------------------------------------------------
# Missing Values
# -------------------------------------------------------

print("\nChecking missing values...")

print(df.isnull().sum())

print("\nRemoving missing values...")

df = df.ffill()

df = df.bfill()

print("Missing values handled.")

# -------------------------------------------------------
# Feature Selection
# -------------------------------------------------------

print("\nSelecting features...")

FEATURES = [
    "internal_temperature_c",
    "internal_humidity_pct",
    "co2_ppm",
    "hive_weight_kg",
    "external_temperature_c",
    "external_humidity_pct",
    "rainfall_mm_hour",
    "wind_speed_mps",
]

TARGET = "swarming_label_next_72h"

X = df[FEATURES].copy()

y = df[TARGET].copy()

print("Features selected.")

print(X.columns)

# -------------------------------------------------------
# Encode Target
# -------------------------------------------------------

print("\nEncoding labels...")

encoder = LabelEncoder()

y = encoder.fit_transform(y)

joblib.dump(encoder, os.path.join(OUTPUT_FOLDER, "label_encoder.pkl"))

print("Label encoder saved.")

# -------------------------------------------------------
# Feature Scaling
# -------------------------------------------------------

print("\nScaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, os.path.join(OUTPUT_FOLDER, "scaler.pkl"))

print("Scaler saved.")

# -------------------------------------------------------
# Train/Test Split
# -------------------------------------------------------

print("\nCreating train/test split...")

split_index = int(len(X_scaled) * 0.80)

X_train = X_scaled[:split_index]

X_test = X_scaled[split_index:]

y_train = y[:split_index]

y_test = y[split_index:]

print("Training Samples :", len(X_train))

print("Testing Samples :", len(X_test))

# -------------------------------------------------------
# Save Processed Data
# -------------------------------------------------------

print("\nSaving processed datasets...")

pd.DataFrame(X_train, columns=FEATURES).to_csv(
    os.path.join(OUTPUT_FOLDER, "X_train.csv"), index=False
)

pd.DataFrame(X_test, columns=FEATURES).to_csv(
    os.path.join(OUTPUT_FOLDER, "X_test.csv"), index=False
)

pd.DataFrame(y_train, columns=["target"]).to_csv(
    os.path.join(OUTPUT_FOLDER, "y_train.csv"), index=False
)

pd.DataFrame(y_test, columns=["target"]).to_csv(
    os.path.join(OUTPUT_FOLDER, "y_test.csv"), index=False
)

print("Processed data saved successfully.")

# -------------------------------------------------------
# Summary
# -------------------------------------------------------

print("\n" + "=" * 60)

print("PREPROCESSING COMPLETED SUCCESSFULLY")

print("=" * 60)

print("\nGenerated Files")

print("--------------------------------")

print("outputs/model_training/X_train.csv")

print("outputs/model_training/X_test.csv")

print("outputs/model_training/y_train.csv")

print("outputs/model_training/y_test.csv")

print("outputs/model_training/scaler.pkl")

print("outputs/model_training/label_encoder.pkl")

print("\nReady for Step 2 (PELT Feature Generation)")
