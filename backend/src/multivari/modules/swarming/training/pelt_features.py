# """
# =========================================================
# Honey Bee Swarming Prediction
# PELT Feature Engineering Module
# =========================================================

# Purpose:
# - Detect hive behaviour change points using PELT
# - Generate temporal change-point features
# - Save enhanced dataset for ML models

# =========================================================
# """


# import os
# import numpy as np
# import pandas as pd
# import ruptures as rpt

# from sklearn.preprocessing import StandardScaler

# from .config import *


# print("=" * 60)
# print("PELT FEATURE ENGINEERING")
# print("=" * 60)


# # -----------------------------------------------------
# # Load Dataset
# # -----------------------------------------------------

# print("\nLoading dataset...")


# df = pd.read_csv(DATASET_PATH)


# # Convert timestamp

# df[TIMESTAMP_COLUMN] = pd.to_datetime(
#     df[TIMESTAMP_COLUMN]
# )


# # Sort time-series correctly

# df = df.sort_values(
#     [
#         HIVE_COLUMN,
#         TIMESTAMP_COLUMN
#     ]
# )


# df.reset_index(
#     drop=True,
#     inplace=True
# )


# print("Dataset Loaded Successfully")

# print("Rows :", len(df))

# print(
#     "Hives :",
#     df[HIVE_COLUMN].nunique()
# )


# # -----------------------------------------------------
# # Missing value handling (FIXED)
# # -----------------------------------------------------

# print("\nHandling missing values...")

# # FIX: Use ffill() and bfill() instead of fillna(method=...)
# df[PELT_COLUMNS] = df[PELT_COLUMNS].ffill()
# df[PELT_COLUMNS] = df[PELT_COLUMNS].bfill()

# # If there are still NaN values, fill with column means
# if df[PELT_COLUMNS].isnull().any().any():
#     print("Filling remaining NaN with column means...")
#     df[PELT_COLUMNS] = df[PELT_COLUMNS].fillna(df[PELT_COLUMNS].mean())


# # -----------------------------------------------------
# # Create PELT columns
# # -----------------------------------------------------

# df["breakpoint"] = 0

# df["days_since_breakpoint"] = 0

# df["breakpoint_density"] = 0

# df["segment_duration"] = 0


# # -----------------------------------------------------
# # Process each hive separately
# # -----------------------------------------------------

# hives = df[HIVE_COLUMN].unique()


# print("\nProcessing hives...")


# for index, hive in enumerate(hives):


#     print(
#         f"{index+1}/{len(hives)} Processing {hive}"
#     )


#     hive_indices = df.index[
#         df[HIVE_COLUMN] == hive
#     ]


#     hive_data = df.loc[
#         hive_indices
#     ].copy()


#     # -------------------------------
#     # Prepare signal
#     # -------------------------------


#     signal = hive_data[
#         PELT_COLUMNS
#     ].values


#     if len(signal) < 50:

#         print(f"  Skipping {hive}: Not enough data points ({len(signal)})")
#         continue


#     # -------------------------------
#     # Scale PELT features
#     # -------------------------------


#     scaler = StandardScaler()

#     signal_scaled = scaler.fit_transform(
#         signal
#     )


#     # -------------------------------
#     # Run PELT
#     # -------------------------------


#     algorithm = rpt.Pelt(
#         model="l2"
#     )


#     algorithm.fit(
#         signal_scaled
#     )


#     change_points = algorithm.predict(
#         pen=10
#     )


#     # remove final point
#     if len(change_points) > 0:

#         change_points = change_points[:-1]


#     local_length = len(hive_data)


#     # -------------------------------
#     # Breakpoint feature
#     # -------------------------------


#     breakpoint_array = np.zeros(
#         local_length
#     )


#     for point in change_points:


#         if point < local_length:

#             breakpoint_array[point] = 1


#     df.loc[
#         hive_indices,
#         "breakpoint"
#     ] = breakpoint_array


#     # -------------------------------
#     # Days since breakpoint
#     # -------------------------------


#     days_since = []

#     last_change = 0


#     for i,value in enumerate(
#         breakpoint_array
#     ):


#         if value == 1:

#             last_change = i


#         days_since.append(
#             i-last_change
#         )


#     df.loc[
#         hive_indices,
#         "days_since_breakpoint"
#     ] = days_since


#     # -------------------------------
#     # Breakpoint density
#     # -------------------------------


#     density = (
#         pd.Series(
#             breakpoint_array
#         )
#         .rolling(
#             window=24,
#             min_periods=1
#         )
#         .sum()
#         .values
#     )


#     df.loc[
#         hive_indices,
#         "breakpoint_density"
#     ] = density


#     # -------------------------------
#     # Segment duration
#     # -------------------------------


#     duration=[]

#     counter=0


#     for value in breakpoint_array:


#         if value == 1:

#             counter=0


#         counter +=1

#         duration.append(
#             counter
#         )


#     df.loc[
#         hive_indices,
#         "segment_duration"
#     ] = duration


# print("\nPELT processing completed")


# # -----------------------------------------------------
# # Save enhanced dataset
# # -----------------------------------------------------


# OUTPUT_FILE = os.path.join(
#     OUTPUT_FOLDER,
#     "hive_data_with_pelt.csv"
# )


# # Create output folder if it doesn't exist
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# df.to_csv(
#     OUTPUT_FILE,
#     index=False
# )


# print("\nSaved File:")

# print(
#     OUTPUT_FILE
# )


# print("\nGenerated PELT Features:")

# print(
#     [
#         "breakpoint",
#         "days_since_breakpoint",
#         "breakpoint_density",
#         "segment_duration"
#     ]
# )


# print("\nPreview:")

# print(
#     df[
#         [
#         HIVE_COLUMN,
#         TIMESTAMP_COLUMN,
#         "breakpoint",
#         "days_since_breakpoint",
#         "breakpoint_density",
#         "segment_duration"
#         ]
#     ].head()
# )

# import json


# # ---------------------------------------------
# # Save PELT Summary Metrics
# # ---------------------------------------------

# pelt_metrics = {


#     "total_hives_processed":
#         int(df[HIVE_COLUMN].nunique()),


#     "total_records":
#         int(len(df)),


#     "total_change_points":
#         int(df["breakpoint"].sum()),


#     "generated_features":[

#         "breakpoint",

#         "days_since_breakpoint",

#         "breakpoint_density",

#         "segment_duration"

#     ]

# }


# PELT_JSON = os.path.join(

#     OUTPUT_FOLDER,

#     "pelt_metrics.json"

# )


# with open(
#     PELT_JSON,
#     "w"
# ) as file:

#     json.dump(
#         pelt_metrics,
#         file,
#         indent=4
#     )


# print("\nPELT Metrics Saved")

# print(PELT_JSON)

# print("\n" + "="*60)

# print(
#     "PELT FEATURE ENGINEERING COMPLETED"
# )

# print("="*60)

# """
# =========================================================
# Honey Bee Swarming Prediction
# PELT Feature Engineering Module (UPDATED)
# =========================================================

# Purpose:
# - Detect hive behaviour change points using PELT (Multivariate)
# - Detect per-variable change points (NEW)
# - Generate temporal change-point features
# - Generate alignment features (NEW)
# - Save enhanced dataset for ML models

# =========================================================
# """


# import os
# import numpy as np
# import pandas as pd
# import ruptures as rpt
# import json
# from sklearn.preprocessing import StandardScaler

# from .config import *


# print("=" * 60)
# print("PELT FEATURE ENGINEERING (UPDATED)")
# print("=" * 60)


# # -----------------------------------------------------
# # Load Dataset
# # -----------------------------------------------------

# print("\nLoading dataset...")


# df = pd.read_csv(DATASET_PATH)


# # Convert timestamp

# df[TIMESTAMP_COLUMN] = pd.to_datetime(
#     df[TIMESTAMP_COLUMN]
# )


# # Sort time-series correctly

# df = df.sort_values(
#     [
#         HIVE_COLUMN,
#         TIMESTAMP_COLUMN
#     ]
# )


# df.reset_index(
#     drop=True,
#     inplace=True
# )


# print("Dataset Loaded Successfully")

# print("Rows :", len(df))

# print(
#     "Hives :",
#     df[HIVE_COLUMN].nunique()
# )


# # -----------------------------------------------------
# # Missing value handling (FIXED)
# # -----------------------------------------------------

# print("\nHandling missing values...")

# # FIX: Use ffill() and bfill() instead of fillna(method=...)
# df[PELT_COLUMNS] = df[PELT_COLUMNS].ffill()
# df[PELT_COLUMNS] = df[PELT_COLUMNS].bfill()

# # If there are still NaN values, fill with column means
# if df[PELT_COLUMNS].isnull().any().any():
#     print("Filling remaining NaN with column means...")
#     df[PELT_COLUMNS] = df[PELT_COLUMNS].fillna(df[PELT_COLUMNS].mean())


# # -----------------------------------------------------
# # Create PELT columns (Existing + New Per-Variable)
# # -----------------------------------------------------

# # Existing columns
# df["breakpoint"] = 0
# df["days_since_breakpoint"] = 0
# df["breakpoint_density"] = 0
# df["segment_duration"] = 0

# # New Per-Variable columns
# VARIABLE_NAMES = ["temp", "hum", "co2", "weight"]

# for var in VARIABLE_NAMES:
#     df[f"breakpoint_{var}"] = 0
#     df[f"days_since_breakpoint_{var}"] = 0
#     df[f"breakpoint_density_{var}"] = 0

# # New Alignment columns
# df["alignment_count"] = 0
# df["all_aligned"] = 0
# df["majority_aligned"] = 0
# df["alignment_ratio"] = 0.0

# # Pairwise alignments
# PAIRS = [("temp", "hum"), ("temp", "co2"), ("temp", "weight"),
#          ("hum", "co2"), ("hum", "weight"), ("co2", "weight")]

# for v1, v2 in PAIRS:
#     df[f"aligned_{v1}_{v2}"] = 0


# # -----------------------------------------------------
# # Process each hive separately
# # -----------------------------------------------------

# hives = df[HIVE_COLUMN].unique()


# print("\nProcessing hives...")


# for index, hive in enumerate(hives):


#     print(
#         f"{index+1}/{len(hives)} Processing {hive}"
#     )


#     hive_indices = df.index[
#         df[HIVE_COLUMN] == hive
#     ]


#     hive_data = df.loc[
#         hive_indices
#     ].copy()


#     if len(hive_data) < 50:
#         print(f"  Skipping {hive}: Not enough data points ({len(hive_data)})")
#         continue


#     local_length = len(hive_data)


#     # =========================================================
#     # PART 1: MULTIVARIATE PELT (Existing)
#     # =========================================================

#     # Prepare signal
#     signal = hive_data[PELT_COLUMNS].values

#     # Scale PELT features
#     scaler = StandardScaler()
#     signal_scaled = scaler.fit_transform(signal)

#     # Run PELT
#     algorithm = rpt.Pelt(model="l2")
#     algorithm.fit(signal_scaled)
#     change_points = algorithm.predict(pen=10)

#     # remove final point
#     if len(change_points) > 0:
#         change_points = change_points[:-1]


#     # -------------------------------
#     # Breakpoint feature
#     # -------------------------------

#     breakpoint_array = np.zeros(local_length)
#     for point in change_points:
#         if point < local_length:
#             breakpoint_array[point] = 1

#     df.loc[hive_indices, "breakpoint"] = breakpoint_array


#     # -------------------------------
#     # Days since breakpoint
#     # -------------------------------

#     days_since = []
#     last_change = 0
#     for i, value in enumerate(breakpoint_array):
#         if value == 1:
#             last_change = i
#         days_since.append(i - last_change)

#     df.loc[hive_indices, "days_since_breakpoint"] = days_since


#     # -------------------------------
#     # Breakpoint density
#     # -------------------------------

#     density = (
#         pd.Series(breakpoint_array)
#         .rolling(window=24, min_periods=1)
#         .sum()
#         .values
#     )

#     df.loc[hive_indices, "breakpoint_density"] = density


#     # -------------------------------
#     # Segment duration
#     # -------------------------------

#     duration = []
#     counter = 0
#     for value in breakpoint_array:
#         if value == 1:
#             counter = 0
#         counter += 1
#         duration.append(counter)

#     df.loc[hive_indices, "segment_duration"] = duration


#     # =========================================================
#     # PART 2: PER-VARIABLE PELT (NEW)
#     # =========================================================

#     for var_idx, var_name in enumerate(PELT_COLUMNS):
#         var_short = VARIABLE_NAMES[var_idx]

#         # Get single variable signal
#         single_signal = hive_data[var_name].values.reshape(-1, 1)

#         # Scale
#         scaler_single = StandardScaler()
#         signal_scaled_single = scaler_single.fit_transform(single_signal)

#         # Run PELT
#         algorithm_single = rpt.Pelt(model="l2")
#         algorithm_single.fit(signal_scaled_single)
#         change_points_single = algorithm_single.predict(pen=10)

#         if len(change_points_single) > 0:
#             change_points_single = change_points_single[:-1]

#         # Breakpoint array for this variable
#         bp_array_single = np.zeros(local_length)
#         for point in change_points_single:
#             if point < local_length:
#                 bp_array_single[point] = 1

#         df.loc[hive_indices, f"breakpoint_{var_short}"] = bp_array_single

#         # Days since breakpoint
#         days_since_single = []
#         last_change_single = 0
#         for i, val in enumerate(bp_array_single):
#             if val == 1:
#                 last_change_single = i
#             days_since_single.append(i - last_change_single)

#         df.loc[hive_indices, f"days_since_breakpoint_{var_short}"] = days_since_single

#         # Density
#         density_single = (
#             pd.Series(bp_array_single)
#             .rolling(window=24, min_periods=1)
#             .sum()
#             .values
#         )

#         df.loc[hive_indices, f"breakpoint_density_{var_short}"] = density_single


#     # =========================================================
#     # PART 3: ALIGNMENT FEATURES (NEW)
#     # =========================================================

#     # Get all breakpoint columns
#     temp_bp = df.loc[hive_indices, "breakpoint_temp"].values
#     hum_bp = df.loc[hive_indices, "breakpoint_hum"].values
#     co2_bp = df.loc[hive_indices, "breakpoint_co2"].values
#     weight_bp = df.loc[hive_indices, "breakpoint_weight"].values

#     # Alignment count (how many variables broke at same time)
#     alignment_count = temp_bp + hum_bp + co2_bp + weight_bp
#     df.loc[hive_indices, "alignment_count"] = alignment_count

#     # All aligned (all 4 breakpoints at same time)
#     all_aligned = (alignment_count == 4).astype(int)
#     df.loc[hive_indices, "all_aligned"] = all_aligned

#     # Majority aligned (3 or more variables)
#     majority_aligned = (alignment_count >= 3).astype(int)
#     df.loc[hive_indices, "majority_aligned"] = majority_aligned

#     # Alignment ratio
#     alignment_ratio = alignment_count / 4
#     df.loc[hive_indices, "alignment_ratio"] = alignment_ratio

#     # Pairwise alignments
#     pairs = {
#         "temp_hum": (temp_bp, hum_bp),
#         "temp_co2": (temp_bp, co2_bp),
#         "temp_weight": (temp_bp, weight_bp),
#         "hum_co2": (hum_bp, co2_bp),
#         "hum_weight": (hum_bp, weight_bp),
#         "co2_weight": (co2_bp, weight_bp),
#     }

#     for pair_name, (bp1, bp2) in pairs.items():
#         aligned = ((bp1 == 1) & (bp2 == 1)).astype(int)
#         df.loc[hive_indices, f"aligned_{pair_name}"] = aligned


# print("\nPELT processing completed")


# # -----------------------------------------------------
# # Save enhanced dataset
# # -----------------------------------------------------


# OUTPUT_FILE = os.path.join(
#     OUTPUT_FOLDER,
#     "hive_data_with_pelt.csv"
# )


# # Create output folder if it doesn't exist
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# df.to_csv(
#     OUTPUT_FILE,
#     index=False
# )


# print("\nSaved File:")

# print(
#     OUTPUT_FILE
# )


# print("\nGenerated PELT Features:")

# print("  Existing Features:")
# print("    - breakpoint")
# print("    - days_since_breakpoint")
# print("    - breakpoint_density")
# print("    - segment_duration")

# print("\n  New Per-Variable Features:")
# for var in VARIABLE_NAMES:
#     print(f"    - breakpoint_{var}")
#     print(f"    - days_since_breakpoint_{var}")
#     print(f"    - breakpoint_density_{var}")

# print("\n  New Alignment Features:")
# print("    - alignment_count")
# print("    - all_aligned")
# print("    - majority_aligned")
# print("    - alignment_ratio")
# for v1, v2 in PAIRS:
#     print(f"    - aligned_{v1}_{v2}")


# print("\nPreview:")

# print(
#     df[
#         [
#         HIVE_COLUMN,
#         TIMESTAMP_COLUMN,
#         "breakpoint",
#         "days_since_breakpoint",
#         "breakpoint_density",
#         "segment_duration",
#         "breakpoint_temp",
#         "breakpoint_hum",
#         "breakpoint_co2",
#         "breakpoint_weight",
#         "alignment_count",
#         "all_aligned"
#         ]
#     ].head(10)
# )


# # ---------------------------------------------
# # Save PELT Summary Metrics
# # ---------------------------------------------

# pelt_metrics = {
#     "total_hives_processed": int(df[HIVE_COLUMN].nunique()),
#     "total_records": int(len(df)),
#     "total_change_points": int(df["breakpoint"].sum()),
#     "generated_features": {
#         "existing": [
#             "breakpoint",
#             "days_since_breakpoint",
#             "breakpoint_density",
#             "segment_duration"
#         ],
#         "per_variable": [
#             f"breakpoint_{var}" for var in VARIABLE_NAMES
#         ] + [
#             f"days_since_breakpoint_{var}" for var in VARIABLE_NAMES
#         ] + [
#             f"breakpoint_density_{var}" for var in VARIABLE_NAMES
#         ],
#         "alignment": [
#             "alignment_count",
#             "all_aligned",
#             "majority_aligned",
#             "alignment_ratio"
#         ] + [
#             f"aligned_{v1}_{v2}" for v1, v2 in PAIRS
#         ]
#     },
#     "total_features": 4 + (3 * 4) + 4 + 6  # existing + per-var + alignment + pairwise
# }


# PELT_JSON = os.path.join(
#     OUTPUT_FOLDER,
#     "pelt_metrics.json"
# )


# with open(PELT_JSON, "w") as file:
#     json.dump(pelt_metrics, file, indent=4)


# print("\nPELT Metrics Saved")
# print(PELT_JSON)

# print(f"\nTotal PELT Features Created: {pelt_metrics['total_features']}")

# print("\n" + "="*60)
# print("PELT FEATURE ENGINEERING COMPLETED")
# print("="*60)
"""
=========================================================
Honey Bee Swarming Prediction
PELT Feature Engineering Module (UPDATED)
=========================================================

Purpose:
- Detect hive behaviour change points using PELT (Multivariate)
- Detect per-variable change points (NEW)
- Generate temporal change-point features
- Generate alignment features (NEW)
- Add month and season features
- Save enhanced dataset for ML models

=========================================================
"""

import json
import os

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.preprocessing import StandardScaler

from .config import *

print("=" * 60)
print("PELT FEATURE ENGINEERING (UPDATED)")
print("=" * 60)


# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

# Convert timestamp
df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])

# Sort time-series correctly
df = df.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN])
df.reset_index(drop=True, inplace=True)

print("Dataset Loaded Successfully")
print("Rows :", len(df))
print("Hives :", df[HIVE_COLUMN].nunique())


# -----------------------------------------------------
# Missing value handling
# -----------------------------------------------------

print("\nHandling missing values...")

df[PELT_COLUMNS] = df[PELT_COLUMNS].ffill()
df[PELT_COLUMNS] = df[PELT_COLUMNS].bfill()

if df[PELT_COLUMNS].isnull().any().any():
    print("Filling remaining NaN with column means...")
    df[PELT_COLUMNS] = df[PELT_COLUMNS].fillna(df[PELT_COLUMNS].mean())


# -----------------------------------------------------
# Create PELT columns (Existing + New Per-Variable)
# -----------------------------------------------------

# Existing columns
df["breakpoint"] = 0
df["days_since_breakpoint"] = 0
df["breakpoint_density"] = 0
df["segment_duration"] = 0

# New Per-Variable columns
VARIABLE_NAMES = ["temp", "hum", "co2", "weight"]

for var in VARIABLE_NAMES:
    df[f"breakpoint_{var}"] = 0
    df[f"days_since_breakpoint_{var}"] = 0
    df[f"breakpoint_density_{var}"] = 0

# New Alignment columns
df["alignment_count"] = 0
df["all_aligned"] = 0
df["majority_aligned"] = 0
df["alignment_ratio"] = 0.0

# Pairwise alignments
PAIRS = [
    ("temp", "hum"),
    ("temp", "co2"),
    ("temp", "weight"),
    ("hum", "co2"),
    ("hum", "weight"),
    ("co2", "weight"),
]

for v1, v2 in PAIRS:
    df[f"aligned_{v1}_{v2}"] = 0


# -----------------------------------------------------
# Process each hive separately
# -----------------------------------------------------

hives = df[HIVE_COLUMN].unique()

print("\nProcessing hives...")

for index, hive in enumerate(hives):
    print(f"{index + 1}/{len(hives)} Processing {hive}")

    hive_indices = df.index[df[HIVE_COLUMN] == hive]
    hive_data = df.loc[hive_indices].copy()

    if len(hive_data) < 50:
        print(f"  Skipping {hive}: Not enough data points ({len(hive_data)})")
        continue

    local_length = len(hive_data)

    # =========================================================
    # PART 1: MULTIVARIATE PELT (Existing)
    # =========================================================

    # Prepare signal
    signal = hive_data[PELT_COLUMNS].values

    # Scale PELT features
    scaler = StandardScaler()
    signal_scaled = scaler.fit_transform(signal)

    # Run PELT
    algorithm = rpt.Pelt(model="l2")
    algorithm.fit(signal_scaled)
    change_points = algorithm.predict(pen=10)

    # remove final point
    if len(change_points) > 0:
        change_points = change_points[:-1]

    # -------------------------------
    # Breakpoint feature
    # -------------------------------

    breakpoint_array = np.zeros(local_length)
    for point in change_points:
        if point < local_length:
            breakpoint_array[point] = 1

    df.loc[hive_indices, "breakpoint"] = breakpoint_array

    # -------------------------------
    # Days since breakpoint
    # -------------------------------

    days_since = []
    last_change = 0
    for i, value in enumerate(breakpoint_array):
        if value == 1:
            last_change = i
        days_since.append(i - last_change)

    df.loc[hive_indices, "days_since_breakpoint"] = days_since

    # -------------------------------
    # Breakpoint density
    # -------------------------------

    density = pd.Series(breakpoint_array).rolling(window=24, min_periods=1).sum().values

    df.loc[hive_indices, "breakpoint_density"] = density

    # -------------------------------
    # Segment duration
    # -------------------------------

    duration = []
    counter = 0
    for value in breakpoint_array:
        if value == 1:
            counter = 0
        counter += 1
        duration.append(counter)

    df.loc[hive_indices, "segment_duration"] = duration

    # =========================================================
    # PART 2: PER-VARIABLE PELT (NEW)
    # =========================================================

    for var_idx, var_name in enumerate(PELT_COLUMNS):
        var_short = VARIABLE_NAMES[var_idx]

        # Get single variable signal
        single_signal = hive_data[var_name].values.reshape(-1, 1)

        # Scale
        scaler_single = StandardScaler()
        signal_scaled_single = scaler_single.fit_transform(single_signal)

        # Run PELT
        algorithm_single = rpt.Pelt(model="l2")
        algorithm_single.fit(signal_scaled_single)
        change_points_single = algorithm_single.predict(pen=10)

        if len(change_points_single) > 0:
            change_points_single = change_points_single[:-1]

        # Breakpoint array for this variable
        bp_array_single = np.zeros(local_length)
        for point in change_points_single:
            if point < local_length:
                bp_array_single[point] = 1

        df.loc[hive_indices, f"breakpoint_{var_short}"] = bp_array_single

        # Days since breakpoint
        days_since_single = []
        last_change_single = 0
        for i, val in enumerate(bp_array_single):
            if val == 1:
                last_change_single = i
            days_since_single.append(i - last_change_single)

        df.loc[hive_indices, f"days_since_breakpoint_{var_short}"] = days_since_single

        # Density
        density_single = pd.Series(bp_array_single).rolling(window=24, min_periods=1).sum().values

        df.loc[hive_indices, f"breakpoint_density_{var_short}"] = density_single

    # =========================================================
    # PART 3: ALIGNMENT FEATURES (NEW)
    # =========================================================

    # Get all breakpoint columns
    temp_bp = df.loc[hive_indices, "breakpoint_temp"].values
    hum_bp = df.loc[hive_indices, "breakpoint_hum"].values
    co2_bp = df.loc[hive_indices, "breakpoint_co2"].values
    weight_bp = df.loc[hive_indices, "breakpoint_weight"].values

    # Alignment count (how many variables broke at same time)
    alignment_count = temp_bp + hum_bp + co2_bp + weight_bp
    df.loc[hive_indices, "alignment_count"] = alignment_count

    # All aligned (all 4 breakpoints at same time)
    all_aligned = (alignment_count == 4).astype(int)
    df.loc[hive_indices, "all_aligned"] = all_aligned

    # Majority aligned (3 or more variables)
    majority_aligned = (alignment_count >= 3).astype(int)
    df.loc[hive_indices, "majority_aligned"] = majority_aligned

    # Alignment ratio
    alignment_ratio = alignment_count / 4
    df.loc[hive_indices, "alignment_ratio"] = alignment_ratio

    # Pairwise alignments
    pairs = {
        "temp_hum": (temp_bp, hum_bp),
        "temp_co2": (temp_bp, co2_bp),
        "temp_weight": (temp_bp, weight_bp),
        "hum_co2": (hum_bp, co2_bp),
        "hum_weight": (hum_bp, weight_bp),
        "co2_weight": (co2_bp, weight_bp),
    }

    for pair_name, (bp1, bp2) in pairs.items():
        aligned = ((bp1 == 1) & (bp2 == 1)).astype(int)
        df.loc[hive_indices, f"aligned_{pair_name}"] = aligned


print("\nPELT processing completed")


# -----------------------------------------------------
# SAVE ENHANCED DATASET
# -----------------------------------------------------

OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "hive_data_with_pelt.csv")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved File:")
print(OUTPUT_FILE)


# -----------------------------------------------------
# ADD MONTH AND SEASON FEATURES
# -----------------------------------------------------

print("\nAdding Month and Season Features...")

df["month"] = df[TIMESTAMP_COLUMN].dt.month
df["season"] = df[TIMESTAMP_COLUMN].dt.quarter

print("  Added: month, season")


# -----------------------------------------------------
# PRINT GENERATED FEATURES
# -----------------------------------------------------

print("\nGenerated PELT Features:")

print("  Existing Features:")
print("    - breakpoint")
print("    - days_since_breakpoint")
print("    - breakpoint_density")
print("    - segment_duration")

print("\n  New Per-Variable Features:")
for var in VARIABLE_NAMES:
    print(f"    - breakpoint_{var}")
    print(f"    - days_since_breakpoint_{var}")
    print(f"    - breakpoint_density_{var}")

print("\n  New Alignment Features:")
print("    - alignment_count")
print("    - all_aligned")
print("    - majority_aligned")
print("    - alignment_ratio")
for v1, v2 in PAIRS:
    print(f"    - aligned_{v1}_{v2}")

print("\n  ✅ New Temporal Features:")
print("    - month")
print("    - season")


# -----------------------------------------------------
# PREVIEW
# -----------------------------------------------------

print("\nPreview:")

print(
    df[
        [
            HIVE_COLUMN,
            TIMESTAMP_COLUMN,
            "breakpoint",
            "days_since_breakpoint",
            "breakpoint_density",
            "segment_duration",
            "breakpoint_temp",
            "breakpoint_hum",
            "breakpoint_co2",
            "breakpoint_weight",
            "alignment_count",
            "all_aligned",
            "month",
            "season",
        ]
    ].head(10)
)


# -----------------------------------------------------
# SAVE PELT SUMMARY METRICS
# -----------------------------------------------------

pelt_metrics = {
    "total_hives_processed": int(df[HIVE_COLUMN].nunique()),
    "total_records": len(df),
    "total_change_points": int(df["breakpoint"].sum()),
    "generated_features": {
        "existing": [
            "breakpoint",
            "days_since_breakpoint",
            "breakpoint_density",
            "segment_duration",
        ],
        "per_variable": [f"breakpoint_{var}" for var in VARIABLE_NAMES]
        + [f"days_since_breakpoint_{var}" for var in VARIABLE_NAMES]
        + [f"breakpoint_density_{var}" for var in VARIABLE_NAMES],
        "alignment": ["alignment_count", "all_aligned", "majority_aligned", "alignment_ratio"]
        + [f"aligned_{v1}_{v2}" for v1, v2 in PAIRS],
        "temporal": ["month", "season"],
    },
    "total_features": 4 + (3 * 4) + 4 + 6 + 2,  # existing + per-var + alignment + temporal
}

PELT_JSON = os.path.join(OUTPUT_FOLDER, "pelt_metrics.json")

with open(PELT_JSON, "w") as file:
    json.dump(pelt_metrics, file, indent=4)

print("\nPELT Metrics Saved")
print(PELT_JSON)
print(f"\nTotal PELT Features Created: {pelt_metrics['total_features']}")

print("\n" + "=" * 60)
print("PELT FEATURE ENGINEERING COMPLETED")
print("=" * 60)
