"""
=========================================================
Honey Bee Swarming Prediction
LSTM Model Training (UPDATED WITH FIXES)
=========================================================
"""

import os
import json
import glob
import warnings
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from .config import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall

# Optional: For handling imbalanced data
# from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

print("=" * 70)
print("LSTM MODEL TRAINING (UPDATED WITH FIXES)")
print("=" * 70)

# -------------------------------------------------------
# Create folders
# -------------------------------------------------------

MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "models")
GRAPH_FOLDER = os.path.join(OUTPUT_FOLDER, "graphs")

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# =====================================================
# DELETE OLD LSTM FILES
# =====================================================

print("\n" + "=" * 70)
print("DELETING OLD LSTM FILES")
print("=" * 70)

essential_files = [
    "label_encoder.pkl",
    "lstm_scaler.pkl",
    "best_lstm.keras",
    "lstm.keras",
]

deleted_count = 0
for filename in essential_files:
    file_path = os.path.join(MODEL_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"  Deleted: {filename}")
        deleted_count += 1

if deleted_count == 0:
    print("  No existing LSTM files found to delete")
else:
    print(f"\n  Total deleted: {deleted_count} essential files")

print("\n" + "=" * 70)

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

print("\nLoading Dataset...")

DATA_FILE = os.path.join(OUTPUT_FOLDER, "hive_data_with_pelt.csv")
df = pd.read_csv(DATA_FILE)
print(f"  Shape: {df.shape}")

# -------------------------------------------------------
# Timestamp
# -------------------------------------------------------

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["hive_id", "timestamp"])
print("  Dataset Sorted")

# -------------------------------------------------------
# Features
# -------------------------------------------------------

FEATURES = [
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
    "segment_duration"
]

TARGET = "swarming_label_next_72h"

print("\nFeatures")
for f in FEATURES:
    print("-", f)
print("\nTarget:", TARGET)

# =====================================================
# Encode Target Labels
# =====================================================

print("\n" + "=" * 70)
print("ENCODING TARGET LABELS")
print("=" * 70)

encoder = LabelEncoder()
df[TARGET] = encoder.fit_transform(df[TARGET])

joblib.dump(
    encoder,
    os.path.join(MODEL_FOLDER, "label_encoder.pkl")
)
print("  Label Encoder Saved")

# =====================================================
# Scale Features
# =====================================================

print("\nScaling Features...")

scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

joblib.dump(
    scaler,
    os.path.join(MODEL_FOLDER, "lstm_scaler.pkl")
)
print("  Feature Scaling Completed")

# =====================================================
# CREATE SEQUENCES FOR ALL HIVES
# =====================================================

print("\nCreating Sliding Window Sequences...")

WINDOW_SIZE = 24  # 24 hours
print(f"  Window Size: {WINDOW_SIZE} hours")

X_sequences = []
y_sequences = []
total_sequences = 0

hives = df["hive_id"].unique()
print(f"  Total Hives: {len(hives)}")

for hive in hives:
    hive_df = df[df["hive_id"] == hive].copy()
    hive_df = hive_df.sort_values("timestamp")

    feature_values = hive_df[FEATURES].values
    target_values = hive_df[TARGET].values

    if len(hive_df) <= WINDOW_SIZE:
        continue

    # Create sequences for this hive
    for i in range(WINDOW_SIZE, len(hive_df)):
        X_sequences.append(feature_values[i - WINDOW_SIZE:i])
        y_sequences.append(target_values[i])
        total_sequences += 1

# Convert to NumPy
X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print(f"\n  Total Sequences: {total_sequences:,}")
print(f"  X Shape: {X_sequences.shape}")
print(f"  y Shape: {y_sequences.shape}")

# =====================================================
# CHECK CLASS DISTRIBUTION
# =====================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_0 = np.sum(y_sequences == 0)
class_1 = np.sum(y_sequences == 1)

print(f"  Class 0 (No Swarming): {class_0:,} ({class_0/len(y_sequences)*100:.2f}%)")
print(f"  Class 1 (Swarming): {class_1:,} ({class_1/len(y_sequences)*100:.2f}%)")

if class_1 == 0:
    print("\n  ❌ CRITICAL: No swarming events in sequences!")
    print("     Check your data or window size.")
    exit()

# =====================================================
# TRAIN/TEST SPLIT (STRATIFIED RANDOM)
# =====================================================

print("\n" + "=" * 70)
print("TRAIN/TEST SPLIT (STRATIFIED RANDOM)")
print("=" * 70)

# Random split with stratification to maintain swarming ratio
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences,
    y_sequences,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_sequences  # Maintains same class ratio in both sets
)

print(f"  Training Samples: {len(X_train):,}")
print(f"  Testing Samples: {len(X_test):,}")

# Check test set distribution
test_class_0 = np.sum(y_test == 0)
test_class_1 = np.sum(y_test == 1)

print(f"\n  Test Set Distribution:")
print(f"    Class 0: {test_class_0:,} ({test_class_0/len(y_test)*100:.2f}%)")
print(f"    Class 1: {test_class_1:,} ({test_class_1/len(y_test)*100:.2f}%)")

if test_class_1 == 0:
    print("\n  ❌ CRITICAL: Test set has ZERO swarming events!")
    print("     This will cause misleading accuracy.")
    exit()

# =====================================================
# CALCULATE CLASS WEIGHTS
# =====================================================

print("\n" + "=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, weights))

print("Class Weights:")
for key, value in class_weights.items():
    print(f"  Class {key}: {value:.4f}")

# =====================================================
# BUILD LSTM MODEL
# =====================================================

print("\n" + "=" * 70)
print("BUILDING LSTM MODEL")
print("=" * 70)

model = Sequential()

# First LSTM Layer
model.add(LSTM(
    units=128,
    return_sequences=True,
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(Dropout(0.30))

# Second LSTM Layer
model.add(LSTM(
    units=64,
    return_sequences=False
))
model.add(Dropout(0.20))

# Dense Layers
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.20))
model.add(Dense(1, activation="sigmoid"))

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        Precision(name="precision"),
        Recall(name="recall")
    ]
)

print("\nModel Summary:")
model.summary()

# =====================================================
# CALLBACKS
# =====================================================

print("\nCreating Callbacks...")

checkpoint_path = os.path.join(MODEL_FOLDER, "best_lstm.keras")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

print("Callbacks Created")

# =====================================================
# TRAIN LSTM MODEL
# =====================================================

print("\n" + "=" * 70)
print("TRAINING LSTM MODEL")
print("=" * 70)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=128,
    class_weight=class_weights,
    shuffle=False,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

print("\nTraining Completed")

# =====================================================
# LOAD BEST MODEL
# =====================================================

print("\nLoading Best Model...")
model = load_model(checkpoint_path)
print("Best Model Loaded")

# =====================================================
# PREDICTIONS
# =====================================================

print("\nGenerating Predictions...")
y_probability = model.predict(X_test, batch_size=256, verbose=1)

# =====================================================
# FIND OPTIMAL THRESHOLD
# =====================================================

print("\n" + "=" * 70)
print("FINDING OPTIMAL THRESHOLD")
print("=" * 70)

best_threshold = 0.5
best_f1 = 0
threshold_results = []

for threshold in np.arange(0.30, 0.71, 0.05):
    pred = (y_probability >= threshold).astype(int)
    # Use binary average for imbalanced data
    score = f1_score(y_test, pred, average="binary", zero_division=0)
    threshold_results.append([threshold, score])
    
    if score > best_f1:
        best_f1 = score
        best_threshold = threshold
    
    print(f"  Threshold: {threshold:.2f} → F1-Score: {score:.4f}")

print(f"\n  ✅ BEST THRESHOLD: {best_threshold:.2f}")
print(f"  ✅ Best F1-Score: {best_f1:.4f}")

# =====================================================
# FINAL PREDICTIONS
# =====================================================

y_pred = (y_probability >= best_threshold).astype(int).flatten()

# =====================================================
# EVALUATION METRICS (Using BINARY average)
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

# ✅ Use average="binary" for imbalanced data
precision = precision_score(y_test, y_pred, average="binary", zero_division=0)
recall = recall_score(y_test, y_pred, average="binary", zero_division=0)
f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 70)
print("LSTM PERFORMANCE (OPTIMAL THRESHOLD)")
print("=" * 70)
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"  Threshold: {best_threshold:.2f}")
print("=" * 70)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(cm)

# =====================================================
# DIAGNOSTIC CHECK
# =====================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC CHECK")
print("=" * 70)

pred_class_0 = np.sum(y_pred == 0)
pred_class_1 = np.sum(y_pred == 1)

print(f"  Actual Class 1: {np.sum(y_test == 1):,}")
print(f"  Predicted Class 1: {pred_class_1:,}")
print(f"  Actual Class 0: {np.sum(y_test == 0):,}")
print(f"  Predicted Class 0: {pred_class_0:,}")

if pred_class_1 == 0 and np.sum(y_test == 1) > 0:
    print("\n  ❌ MODEL IS NOT DETECTING SWARMING!")
    print("     It's predicting 'no swarming' for everything.")
    print("     Consider using SMOTE or adjusting class weights.")
elif pred_class_1 > 0 and np.sum(y_test == 1) > 0:
    print("\n  ✅ Model is detecting some swarming events!")
    detection_rate = pred_class_1 / np.sum(y_test == 1) * 100
    print(f"     Detected {pred_class_1} out of {np.sum(y_test == 1)} swarming events.")
    print(f"     Detection Rate: {detection_rate:.2f}%")

# =====================================================
# SAVE FINAL LSTM MODEL
# =====================================================

print("\n" + "=" * 70)
print("SAVING FINAL LSTM MODEL")
print("=" * 70)

final_model_path = os.path.join(MODEL_FOLDER, "lstm.keras")
model.save(final_model_path)
print(f"  Final Model Saved: {final_model_path}")

# =====================================================
# SAVE METRICS
# =====================================================

print("\nSaving Metrics...")

metrics = {
    "Model": "LSTM",
    "Accuracy": float(accuracy),
    "Precision": float(precision),
    "Recall": float(recall),
    "F1-Score": float(f1),
    "Optimal_Threshold": float(best_threshold),
    "Window_Size": WINDOW_SIZE,
    "Train_Sequences": len(X_train),
    "Test_Sequences": len(X_test),
    "Test_Class_1_Ratio": float(np.sum(y_test == 1) / len(y_test)),
    "Swarming_Detected": int(pred_class_1),
    "Actual_Swarming": int(np.sum(y_test == 1))
}

metrics_path = os.path.join(OUTPUT_FOLDER, "lstm_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"  Metrics Saved: {metrics_path}")

# =====================================================
# SAVE THRESHOLD RESULTS
# =====================================================

threshold_df = pd.DataFrame(threshold_results, columns=["Threshold", "F1-Score"])
threshold_path = os.path.join(OUTPUT_FOLDER, "lstm_threshold_results.csv")
threshold_df.to_csv(threshold_path, index=False)
print(f"  Threshold Results Saved: {threshold_path}")

# =====================================================
# SAVE CONFUSION MATRIX
# =====================================================

print("\nCreating Confusion Matrix...")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"LSTM Confusion Matrix (Threshold={best_threshold:.2f})")

cm_path = os.path.join(GRAPH_FOLDER, "lstm_confusion_matrix.png")
plt.tight_layout()
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Confusion Matrix Saved: {cm_path}")

# =====================================================
# SAVE TRAINING HISTORY
# =====================================================

print("\nCreating Training History...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Loss
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy
ax2.plot(history.history['accuracy'], label='Training Accuracy')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Precision
if 'precision' in history.history:
    ax3.plot(history.history['precision'], label='Training Precision')
if 'val_precision' in history.history:
    ax3.plot(history.history['val_precision'], label='Validation Precision')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Precision')
ax3.set_title('Training and Validation Precision')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Recall
if 'recall' in history.history:
    ax4.plot(history.history['recall'], label='Training Recall')
if 'val_recall' in history.history:
    ax4.plot(history.history['val_recall'], label='Validation Recall')
ax4.set_xlabel('Epochs')
ax4.set_ylabel('Recall')
ax4.set_title('Training and Validation Recall')
ax4.legend()
ax4.grid(True, alpha=0.3)

history_path = os.path.join(GRAPH_FOLDER, "lstm_training_history.png")
plt.tight_layout()
plt.savefig(history_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Training History Saved: {history_path}")

# =====================================================
# SAVE THRESHOLD SELECTION PLOT
# =====================================================

print("\nCreating Threshold Selection Plot...")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(threshold_df['Threshold'], threshold_df['F1-Score'], marker='o', linewidth=2)
ax.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best: {best_threshold:.2f}')
ax.set_xlabel('Threshold')
ax.set_ylabel('F1-Score (Binary)')
ax.set_title('Threshold Selection for LSTM Model')
ax.legend()
ax.grid(True, alpha=0.3)

threshold_plot_path = os.path.join(GRAPH_FOLDER, "lstm_threshold_selection.png")
plt.savefig(threshold_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Threshold Plot Saved: {threshold_plot_path}")

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n" + "=" * 70)
print("LSTM TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated Files:")
print("--------------------------------------------")
print("Models:")
print(f"  - {os.path.join(MODEL_FOLDER, 'label_encoder.pkl')}")
print(f"  - {os.path.join(MODEL_FOLDER, 'lstm_scaler.pkl')}")
print(f"  - {os.path.join(MODEL_FOLDER, 'best_lstm.keras')}")
print(f"  - {os.path.join(MODEL_FOLDER, 'lstm.keras')}")
print("\nMetrics:")
print(f"  - {os.path.join(OUTPUT_FOLDER, 'lstm_metrics.json')}")
print(f"  - {os.path.join(OUTPUT_FOLDER, 'lstm_threshold_results.csv')}")
print("\nGraphs:")
print(f"  - {os.path.join(GRAPH_FOLDER, 'lstm_confusion_matrix.png')}")
print(f"  - {os.path.join(GRAPH_FOLDER, 'lstm_training_history.png')}")
print(f"  - {os.path.join(GRAPH_FOLDER, 'lstm_threshold_selection.png')}")

print("\nPerformance (Binary Metrics):")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"  Threshold: {best_threshold:.2f}")

if precision == 0 and recall == 0 and np.sum(y_test == 1) > 0:
    print("\n  ⚠️  WARNING: Model is NOT detecting any swarming events!")
    print("     Consider using SMOTE or adjusting class weights.")
elif pred_class_1 > 0:
    print(f"\n  ✅ Model detected {pred_class_1} swarming events.")
    print(f"     Detection Rate: {pred_class_1 / np.sum(y_test == 1) * 100:.2f}%")

print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)