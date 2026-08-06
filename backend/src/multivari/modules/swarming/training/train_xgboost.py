# """
# =========================================================
# Honey Bee Swarming Prediction

# XGBoost Training

# Part 1
# =========================================================
# """

# import os
# import json
# import joblib
# import warnings

# import numpy as np
# import pandas as pd

# import matplotlib
# matplotlib.use("Agg")

# import matplotlib.pyplot as plt

# from xgboost import XGBClassifier

# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     classification_report
# )

# from .config import *

# warnings.filterwarnings("ignore")

# print("="*70)
# print("XGBOOST MODEL TRAINING")
# print("="*70)

# # -----------------------------------------------------
# # Output folders
# # -----------------------------------------------------

# MODEL_FOLDER = os.path.join(
#     OUTPUT_FOLDER,
#     "models"
# )

# GRAPH_FOLDER = os.path.join(
#     OUTPUT_FOLDER,
#     "graphs"
# )

# os.makedirs(MODEL_FOLDER, exist_ok=True)
# os.makedirs(GRAPH_FOLDER, exist_ok=True)

# # -----------------------------------------------------
# # Load Dataset
# # -----------------------------------------------------

# print("\nLoading Dataset...")

# DATA_FILE = os.path.join(
#     OUTPUT_FOLDER,
#     "hive_data_with_pelt.csv"
# )

# df = pd.read_csv(DATA_FILE)

# print("Dataset Loaded")

# print(df.shape)

# # -----------------------------------------------------
# # Feature Selection
# # -----------------------------------------------------

# FEATURES = [

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
#     "segment_duration"

# ]

# TARGET = "swarming_label_next_72h"

# X = df[FEATURES]

# y = df[TARGET]

# print("\nFeature Count :", len(FEATURES))
# print("Target :", TARGET)

# # -----------------------------------------------------
# # Encode Labels
# # -----------------------------------------------------

# encoder = LabelEncoder()

# y = encoder.fit_transform(y)

# joblib.dump(

#     encoder,

#     os.path.join(
#         MODEL_FOLDER,
#         "label_encoder.pkl"
#     )

# )

# # -----------------------------------------------------
# # Time Sorting
# # -----------------------------------------------------

# df["timestamp"] = pd.to_datetime(df["timestamp"])

# df = df.sort_values(

#     ["hive_id", "timestamp"]

# )

# X = df[FEATURES]

# y = encoder.transform(df[TARGET])

# # -----------------------------------------------------
# # Train/Test Split
# # -----------------------------------------------------

# split_index = int(

#     len(df) * TRAIN_RATIO

# )

# X_train = X.iloc[:split_index]

# X_test = X.iloc[split_index:]

# y_train = y[:split_index]

# y_test = y[split_index:]

# print("\nTraining Samples :", len(X_train))

# print("Testing Samples :", len(X_test))

# # -----------------------------------------------------
# # Scaling
# # -----------------------------------------------------

# scaler = StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train)

# X_test_scaled = scaler.transform(X_test)

# joblib.dump(

#     scaler,

#     os.path.join(
#         MODEL_FOLDER,
#         "scaler.pkl"
#     )

# )

# print("\nData Preparation Completed")

# print("="*70)


# # =====================================================
# # Calculate Class Weight
# # =====================================================

# print("\nCalculating Class Balance...")

# negative_count = np.sum(y_train == 0)
# positive_count = np.sum(y_train == 1)

# scale_pos_weight = negative_count / positive_count

# print(f"Negative Samples : {negative_count}")
# print(f"Positive Samples : {positive_count}")
# print(f"Scale Pos Weight : {scale_pos_weight:.2f}")

# # =====================================================
# # Train XGBoost Model
# # =====================================================

# print("\n" + "=" * 70)
# print("TRAINING XGBOOST MODEL")
# print("=" * 70)

# xgb_model = XGBClassifier(

#     n_estimators=300,

#     max_depth=6,

#     learning_rate=0.05,

#     subsample=0.8,

#     colsample_bytree=0.8,

#     objective="binary:logistic",

#     eval_metric="logloss",

#     random_state=RANDOM_STATE,

#     scale_pos_weight=scale_pos_weight,

#     n_jobs=-1

# )

# print("\nTraining Model...")

# xgb_model.fit(

#     X_train,

#     y_train

# )

# print("Training Completed!")

# # =====================================================
# # Prediction
# # =====================================================

# print("\nGenerating Predictions...")

# y_pred = xgb_model.predict(X_test)

# print("Prediction Completed!")

# # =====================================================
# # Evaluation
# # =====================================================

# print("\nCalculating Evaluation Metrics...")

# accuracy = accuracy_score(

#     y_test,

#     y_pred

# )

# precision = precision_score(

#     y_test,

#     y_pred,

#     average="weighted",

#     zero_division=0

# )

# recall = recall_score(

#     y_test,

#     y_pred,

#     average="weighted",

#     zero_division=0

# )

# f1 = f1_score(

#     y_test,

#     y_pred,

#     average="weighted",

#     zero_division=0

# )

# print("\n================ MODEL PERFORMANCE ================")

# print(f"Accuracy :  {accuracy:.4f}")
# print(f"Precision:  {precision:.4f}")
# print(f"Recall   :  {recall:.4f}")
# print(f"F1 Score :  {f1:.4f}")

# print("==================================================")

# # =====================================================
# # Classification Report
# # =====================================================

# print("\nClassification Report\n")

# report = classification_report(

#     y_test,

#     y_pred,

#     zero_division=0

# )

# print(report)

# # =====================================================
# # Confusion Matrix
# # =====================================================

# cm = confusion_matrix(

#     y_test,

#     y_pred

# )

# print("\nConfusion Matrix")

# print(cm)

# # =====================================================
# # Store Metrics
# # =====================================================

# xgb_metrics = {

#     "Model": "XGBoost",

#     "Accuracy": float(accuracy),

#     "Precision": float(precision),

#     "Recall": float(recall),

#     "F1-Score": float(f1)

# }

# print("\nMetrics Dictionary Created")

# print(xgb_metrics)

# # =====================================================
# # Save Trained Model
# # =====================================================

# print("\nSaving XGBoost Model...")

# MODEL_PATH = os.path.join(
#     MODEL_FOLDER,
#     "xgboost.pkl"
# )

# joblib.dump(
#     xgb_model,
#     MODEL_PATH
# )

# print("Model Saved Successfully")
# print(MODEL_PATH)


# # =====================================================
# # Save Metrics
# # =====================================================

# print("\nSaving Metrics...")

# METRIC_PATH = os.path.join(
#     OUTPUT_FOLDER,
#     "xgb_metrics.json"
# )

# with open(METRIC_PATH, "w") as file:
#     json.dump(
#         xgb_metrics,
#         file,
#         indent=4
#     )

# print("Metrics Saved")


# # =====================================================
# # Confusion Matrix Plot
# # =====================================================

# plt.figure(figsize=(6,5))

# plt.imshow(cm, cmap="Blues")

# plt.title("XGBoost Confusion Matrix")

# plt.colorbar()

# plt.xticks([0,1],["No Swarm","Swarm"])

# plt.yticks([0,1],["No Swarm","Swarm"])

# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         plt.text(
#             j,
#             i,
#             str(cm[i,j]),
#             ha="center",
#             va="center",
#             color="black"
#         )

# plt.xlabel("Predicted")

# plt.ylabel("Actual")

# plt.tight_layout()

# plt.savefig(
#     os.path.join(
#         GRAPH_FOLDER,
#         "xgb_confusion_matrix.png"
#     )
# )

# plt.close("all")

# print("Confusion Matrix Saved")


# # =====================================================
# # Feature Importance
# # =====================================================

# importance = pd.DataFrame({

#     "Feature": FEATURES,

#     "Importance": xgb_model.feature_importances_

# })

# importance = importance.sort_values(
#     by="Importance",
#     ascending=False
# )

# importance.to_csv(

#     os.path.join(
#         OUTPUT_FOLDER,
#         "xgb_feature_importance.csv"
#     ),

#     index=False

# )

# plt.figure(figsize=(10,6))

# plt.barh(

#     importance["Feature"],

#     importance["Importance"]

# )

# plt.gca().invert_yaxis()

# plt.title("XGBoost Feature Importance")

# plt.tight_layout()

# plt.savefig(

#     os.path.join(
#         GRAPH_FOLDER,
#         "xgb_feature_importance.png"
#     )

# )

# plt.close("all")

# print("Feature Importance Saved")


# print("\n" + "="*70)

# print("XGBOOST TRAINING COMPLETED")

# print("="*70)

# print(f"Accuracy : {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall   : {recall:.4f}")
# print(f"F1 Score : {f1:.4f}")

# print("="*70)


"""
=========================================================
Honey Bee Swarming Prediction
XGBoost Training (UPDATED WITH FIXES)
=========================================================
"""

import os
import json
import glob
import joblib
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from .config import *

warnings.filterwarnings("ignore")

print("=" * 70)
print("XGBOOST MODEL TRAINING (UPDATED)")
print("=" * 70)

# -----------------------------------------------------
# Output folders
# -----------------------------------------------------

MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "models")
GRAPH_FOLDER = os.path.join(OUTPUT_FOLDER, "graphs")

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# =====================================================
# DELETE OLD XGBOOST FILES
# =====================================================

print("\n" + "=" * 70)
print("DELETING OLD XGBOOST FILES")
print("=" * 70)

xgb_files_to_delete = [
    "xgboost.pkl",
    "scaler.pkl",
    "label_encoder.pkl",
]

deleted_count = 0
for filename in xgb_files_to_delete:
    file_path = os.path.join(MODEL_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"  Deleted: {filename}")
        deleted_count += 1

if deleted_count == 0:
    print("  No existing XGBoost files found to delete")
else:
    print(f"\n  Total deleted: {deleted_count} files")

print("\n" + "=" * 70)

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

print("\nLoading Dataset...")

DATA_FILE = os.path.join(OUTPUT_FOLDER, "hive_data_with_pelt.csv")
df = pd.read_csv(DATA_FILE)

print(f"  Shape: {df.shape}")

# -----------------------------------------------------
# Feature Selection
# -----------------------------------------------------

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

print(f"\n  Feature Count: {len(FEATURES)}")
print(f"  Target: {TARGET}")

# -----------------------------------------------------
# Prepare Features and Target
# -----------------------------------------------------

X = df[FEATURES]
y = df[TARGET]

# -----------------------------------------------------
# Encode Labels
# -----------------------------------------------------

print("\nEncoding Target Labels...")

encoder = LabelEncoder()
y = encoder.fit_transform(y)

joblib.dump(
    encoder,
    os.path.join(MODEL_FOLDER, "label_encoder.pkl")
)
print("  Label Encoder Saved")

# -----------------------------------------------------
# Time Sort (For Feature Engineering Only)
# -----------------------------------------------------

print("\nSorting by Time...")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["hive_id", "timestamp"])

X = df[FEATURES]
y = encoder.transform(df[TARGET])

print("  Completed")

# =====================================================
# CHECK CLASS DISTRIBUTION
# =====================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_0 = np.sum(y == 0)
class_1 = np.sum(y == 1)

print(f"  Class 0 (No Swarming): {class_0:,} ({class_0/len(y)*100:.2f}%)")
print(f"  Class 1 (Swarming): {class_1:,} ({class_1/len(y)*100:.2f}%)")

if class_1 == 0:
    print("\n  ❌ CRITICAL: No swarming events in dataset!")
    print("     Cannot train a model without swarming examples.")
    exit()

# =====================================================
# TRAIN/TEST SPLIT (STRATIFIED RANDOM)
# =====================================================

print("\n" + "=" * 70)
print("TRAIN/TEST SPLIT (STRATIFIED RANDOM)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y  # Maintains same class ratio in both sets
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
# SCALE FEATURES
# =====================================================

print("\nScaling Features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(
    scaler,
    os.path.join(MODEL_FOLDER, "scaler.pkl")
)
print("  Scaler Saved")

print("\nData Preparation Completed")
print("=" * 70)

# =====================================================
# CALCULATE CLASS WEIGHT
# =====================================================

print("\n" + "=" * 70)
print("CALCULATING CLASS WEIGHT")
print("=" * 70)

negative_count = np.sum(y_train == 0)
positive_count = np.sum(y_train == 1)
scale_pos_weight = negative_count / positive_count

print(f"  Negative Samples: {negative_count:,}")
print(f"  Positive Samples: {positive_count:,}")
print(f"  Scale Pos Weight: {scale_pos_weight:.2f}")

# =====================================================
# TRAIN XGBOOST MODEL
# =====================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST MODEL")
print("=" * 70)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1
)

print("\nTraining model...")
xgb_model.fit(X_train_scaled, y_train)
print("Training Completed!")

# =====================================================
# PREDICTIONS
# =====================================================

print("\nGenerating Predictions...")
y_pred = xgb_model.predict(X_test_scaled)
print("Prediction Completed!")

# =====================================================
# EVALUATION METRICS (Binary for imbalanced data)
# =====================================================

print("\nCalculating Evaluation Metrics...")

accuracy = accuracy_score(y_test, y_pred)

# ✅ Use binary average for imbalanced data
precision = precision_score(y_test, y_pred, average="binary", zero_division=0)
recall = recall_score(y_test, y_pred, average="binary", zero_division=0)
f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 70)
print("XGBOOST PERFORMANCE (BINARY METRICS)")
print("=" * 70)
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1 Score : {f1:.4f}")
print("=" * 70)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

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
    print("     Consider adjusting scale_pos_weight or using SMOTE.")
elif pred_class_1 > 0 and np.sum(y_test == 1) > 0:
    print("\n  ✅ Model is detecting some swarming events!")
    print(f"     Detected {pred_class_1} out of {np.sum(y_test == 1)} swarming events.")
    detection_rate = pred_class_1 / np.sum(y_test == 1) * 100
    print(f"     Detection Rate: {detection_rate:.2f}%")

# =====================================================
# STORE METRICS
# =====================================================

xgb_metrics = {
    "Model": "XGBoost",
    "Accuracy": float(accuracy),
    "Precision": float(precision),
    "Recall": float(recall),
    "F1-Score": float(f1),
    "Scale_Pos_Weight": float(scale_pos_weight),
    "Test_Class_1_Ratio": float(np.sum(y_test == 1) / len(y_test)),
    "Train_Samples": len(X_train),
    "Test_Samples": len(X_test),
    "Swarming_Detected": int(pred_class_1),
    "Actual_Swarming": int(np.sum(y_test == 1))
}

print("\nMetrics Dictionary Created")
print(xgb_metrics)

# =====================================================
# SAVE TRAINED MODEL
# =====================================================

print("\nSaving XGBoost Model...")

MODEL_PATH = os.path.join(MODEL_FOLDER, "xgboost.pkl")
joblib.dump(xgb_model, MODEL_PATH)
print(f"  Model Saved: {MODEL_PATH}")

# =====================================================
# SAVE METRICS
# =====================================================

print("\nSaving Metrics...")

METRIC_PATH = os.path.join(OUTPUT_FOLDER, "xgb_metrics.json")
with open(METRIC_PATH, "w") as file:
    json.dump(xgb_metrics, file, indent=4)

print(f"  Metrics Saved: {METRIC_PATH}")

# =====================================================
# PLOT CONFUSION MATRIX
# =====================================================

print("\nCreating Confusion Matrix Plot...")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

CONFUSION_PATH = os.path.join(GRAPH_FOLDER, "xgb_confusion_matrix.png")
plt.tight_layout()
plt.savefig(CONFUSION_PATH, dpi=300, bbox_inches='tight')
plt.close()

print(f"  Confusion Matrix Saved: {CONFUSION_PATH}")

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

print("\nCreating Feature Importance Plot...")

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": xgb_model.feature_importances_
})
importance = importance.sort_values(by="Importance", ascending=False)

# Print feature importance
print("\nFeature Importance:")
for idx, row in importance.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(importance["Feature"], importance["Importance"])
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")

FEATURE_PATH = os.path.join(GRAPH_FOLDER, "xgb_feature_importance.png")
plt.tight_layout()
plt.savefig(FEATURE_PATH, dpi=300, bbox_inches='tight')
plt.close()

print(f"  Feature Importance Saved: {FEATURE_PATH}")

# =====================================================
# SAVE CSV FEATURE IMPORTANCE
# =====================================================

importance.to_csv(
    os.path.join(OUTPUT_FOLDER, "xgb_feature_importance.csv"),
    index=False
)
print("  Feature Importance CSV Saved")

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n" + "=" * 70)
print("XGBOOST TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nPerformance (Binary Metrics):")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1 Score : {f1:.4f}")

if precision == 0 and recall == 0 and np.sum(y_test == 1) > 0:
    print("\n  ⚠️  WARNING: Model is NOT detecting any swarming events!")
    print("     Consider adjusting scale_pos_weight or using SMOTE.")

print("\nGenerated Files:")
print("---------------------------------------------")
print("Model:")
print(f"  {MODEL_PATH}")
print("\nMetrics:")
print(f"  {METRIC_PATH}")
print("\nGraphs:")
print(f"  {CONFUSION_PATH}")
print(f"  {FEATURE_PATH}")
print("\nCSV:")
print(f"  {os.path.join(OUTPUT_FOLDER, 'xgb_feature_importance.csv')}")

print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)