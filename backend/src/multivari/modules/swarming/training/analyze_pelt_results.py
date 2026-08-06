# """
# =========================================================
# PELT RESULTS - CLEAN IMAGES FOR PUBLICATION
# (Auto-deletes existing images before generating new ones)
# (Counts ACTUAL swarming events by unique dates - NOT hours)
# (Shows ALL hives in the table with dynamic sizing)
# =========================================================
# """

# import os
# import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from .config import *

# print("=" * 60)
# print("GENERATING CLEAN PELT RESULTS IMAGES")
# print("=" * 60)

# # -----------------------------------------------------
# # Create folders
# # -----------------------------------------------------

# GRAPH_FOLDER = os.path.join(OUTPUT_FOLDER, "graphs")
# os.makedirs(GRAPH_FOLDER, exist_ok=True)

# # -----------------------------------------------------
# # DELETE EXISTING PELT IMAGES
# # -----------------------------------------------------

# print("\n" + "=" * 60)
# print("DELETING EXISTING PELT IMAGES")
# print("=" * 60)

# image_patterns = [
#     "pelt_breakpoint_summary.png",
#     "pelt_breakpoints_barchart.png",
#     "pelt_regime_distribution.png",
#     "pelt_overall_statistics.png",
#     "pelt_heatmap_clean.png",
#     "pelt_swarming_vs_breakpoints.png",
#     "pelt_summary_with_swarming.png",
# ]

# deleted_count = 0
# for pattern in image_patterns:
#     file_path = os.path.join(GRAPH_FOLDER, pattern)
#     if os.path.exists(file_path):
#         os.remove(file_path)
#         print(f"  Deleted: {pattern}")
#         deleted_count += 1

# if deleted_count == 0:
#     print("  No existing images found to delete")
# else:
#     print(f"\n  Total deleted: {deleted_count} images")

# print("\n" + "=" * 60)
# print("GENERATING NEW IMAGES")
# print("=" * 60)

# # -----------------------------------------------------
# # Load data
# # -----------------------------------------------------

# print("\nLoading data...")

# DATA_FILE = os.path.join(OUTPUT_FOLDER, "hive_data_with_pelt.csv")
# df = pd.read_csv(DATA_FILE)

# print(f"  Loaded: {DATA_FILE}")
# print(f"  Records: {len(df):,}")
# print(f"  Hives: {df[HIVE_COLUMN].nunique():,}")

# # -----------------------------------------------------
# # Convert timestamp to date for unique event counting
# # -----------------------------------------------------

# print("\nConverting timestamps to dates...")

# df['date'] = pd.to_datetime(df[TIMESTAMP_COLUMN]).dt.date

# print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
# print(f"  Unique dates: {df['date'].nunique():,}")

# # -----------------------------------------------------
# # Check for swarming labels
# # -----------------------------------------------------

# print("\n" + "=" * 60)
# print("SWARMING EVENTS ANALYSIS")
# print("=" * 60)

# SWARMING_LABEL = TARGET_COLUMN  # "swarming_label_next_72h"
# SWARMING_EVENT_COLUMN = "swarming_event_label"  # Column W

# # Check which columns exist
# if SWARMING_EVENT_COLUMN in df.columns:
#     print(f"\n1. Using '{SWARMING_EVENT_COLUMN}' for event counting...")
    
#     # Count UNIQUE DATES where swarming_event_label = 1
#     swarming_dates = df[df[SWARMING_EVENT_COLUMN] == 1]['date'].unique()
#     total_events = len(swarming_dates)
    
#     print(f"   Total UNIQUE swarming dates: {total_events:,}")
#     print(f"   Total rows with label=1: {df[SWARMING_EVENT_COLUMN].sum():,}")
#     print(f"   Average hours per event: {df[SWARMING_EVENT_COLUMN].sum() / total_events if total_events > 0 else 0:.1f}")
    
#     # Count events per hive
#     events_per_hive = df[df[SWARMING_EVENT_COLUMN] == 1].groupby(HIVE_COLUMN)['date'].nunique().reset_index()
#     events_per_hive.columns = [HIVE_COLUMN, 'swarming_events']
    
#     print(f"\n   Hives with events: {len(events_per_hive)}")
#     print(f"   Total unique events: {events_per_hive['swarming_events'].sum():,}")
    
# elif SWARMING_LABEL in df.columns:
#     print(f"\n1. '{SWARMING_EVENT_COLUMN}' not found. Using '{SWARMING_LABEL}'...")
    
#     # For swarming_label_next_72h, count unique dates with label=1
#     swarming_dates = df[df[SWARMING_LABEL] == 1]['date'].unique()
#     total_events = len(swarming_dates)
    
#     print(f"   Total UNIQUE swarming dates: {total_events:,}")
#     print(f"   Total rows with label=1: {df[SWARMING_LABEL].sum():,}")
    
#     # Count events per hive
#     events_per_hive = df[df[SWARMING_LABEL] == 1].groupby(HIVE_COLUMN)['date'].nunique().reset_index()
#     events_per_hive.columns = [HIVE_COLUMN, 'swarming_events']
    
# else:
#     print("\n  ERROR: No swarming labels found!")
#     events_per_hive = pd.DataFrame({HIVE_COLUMN: [], 'swarming_events': []})

# # -----------------------------------------------------
# # Create Breakpoint Summary with Swarming Events
# # -----------------------------------------------------

# print("\n" + "=" * 60)
# print("CREATING BREAKPOINT SUMMARY")
# print("=" * 60)

# breakpoint_summary = df.groupby(HIVE_COLUMN).agg(
#     total_records=('breakpoint', 'count'),
#     total_breakpoints=('breakpoint', 'sum'),
#     breakpoint_density_avg=('breakpoint_density', 'mean'),
#     max_breakpoint_density=('breakpoint_density', 'max')
# ).reset_index()

# breakpoint_summary['breakpoints_per_100h'] = (
#     breakpoint_summary['total_breakpoints'] / 
#     breakpoint_summary['total_records'] * 100
# )

# # Merge with swarming events (unique counts)
# if len(events_per_hive) > 0:
#     breakpoint_summary = breakpoint_summary.merge(
#         events_per_hive, 
#         on=HIVE_COLUMN, 
#         how='left'
#     )
#     breakpoint_summary['swarming_events'] = breakpoint_summary['swarming_events'].fillna(0).astype(int)
# else:
#     breakpoint_summary['swarming_events'] = 0

# # Sort by breakpoints descending
# breakpoint_summary = breakpoint_summary.sort_values(
#     'total_breakpoints', ascending=False
# )

# # Print summary
# total_events = breakpoint_summary['swarming_events'].sum()
# hives_with_events = len(breakpoint_summary[breakpoint_summary['swarming_events'] > 0])

# print(f"\nBreakpoint Summary Created:")
# print(f"  Total Hives: {len(breakpoint_summary)}")
# print(f"  Total Actual Swarming Events: {total_events:,}")
# print(f"  Hives with Swarming: {hives_with_events}")

# # -----------------------------------------------------
# # 1. BREAKPOINT SUMMARY TABLE - ALL HIVES
# # -----------------------------------------------------

# print("\n1. Creating Breakpoint Summary Table with ALL Hives...")

# # Show ALL hives instead of just top 20
# all_hives = breakpoint_summary

# # Calculate figure height based on number of hives
# num_hives = len(all_hives)
# fig_height = max(8, num_hives * 0.32)  # Dynamic height

# fig, ax = plt.subplots(figsize=(14, fig_height))
# ax.axis('off')

# table_data = all_hives.values.tolist()
# col_labels = ['Hive', 'Records', 'Breakpoints', 'Avg Density', 'Max Density', 'per 100h', 'Swarming Events']

# formatted_data = []
# for row in table_data:
#     formatted_data.append([
#         row[0],
#         f"{row[1]:,}",
#         f"{row[2]:,}",
#         f"{row[3]:.2f}",
#         f"{row[4]:.0f}",
#         f"{row[5]:.2f}",
#         f"{row[6]:,}" if len(row) > 6 else "0"
#     ])

# # Adjust font size based on number of rows
# font_size = max(6, min(9, 14 - num_hives * 0.06))

# table = ax.table(
#     cellText=formatted_data,
#     colLabels=col_labels,
#     cellLoc='center',
#     loc='center',
#     colWidths=[0.10, 0.12, 0.14, 0.14, 0.12, 0.14, 0.14]
# )

# table.auto_set_font_size(False)
# table.set_fontsize(font_size)
# table.scale(1.2, 1.5)

# # Style header
# for j in range(len(col_labels)):
#     table[(0, j)].set_facecolor('#2C3E50')
#     table[(0, j)].set_text_props(color='white', fontweight='bold')

# # Style rows
# for i in range(1, len(formatted_data) + 1):
#     row_data = table_data[i-1]
#     for j in range(len(col_labels)):
#         if i % 2 == 0:
#             table[(i, j)].set_facecolor('#ECF0F1')
#         else:
#             table[(i, j)].set_facecolor('#FFFFFF')
    
#     # Highlight hives with swarming events
#     if len(row_data) > 6 and row_data[6] > 0:
#         for j in range(len(col_labels)):
#             table[(i, j)].set_facecolor('#FFF3E0')
#             if j == len(col_labels) - 1:
#                 if row_data[6] >= 3:
#                     table[(i, j)].set_facecolor('#E74C3C')
#                 elif row_data[6] >= 2:
#                     table[(i, j)].set_facecolor('#F39C12')
#                 else:
#                     table[(i, j)].set_facecolor('#3498DB')
#                 table[(i, j)].set_text_props(color='white', fontweight='bold')

# ax.set_title(f'Table 1: All {num_hives} Hives with Breakpoints and Swarming Events', 
#              fontsize=14, fontweight='bold', pad=20)

# plt.tight_layout()
# SAVE_PATH = os.path.join(GRAPH_FOLDER, "pelt_breakpoint_summary.png")
# plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', facecolor='white')
# plt.close()
# print(f"  Saved: {SAVE_PATH} (All {num_hives} hives)")

# # -----------------------------------------------------
# # 2. BREAKPOINTS BAR CHART WITH SWARMING INDICATORS
# # -----------------------------------------------------

# print("\n2. Creating Breakpoint Bar Chart with Swarming Indicators...")

# fig, ax = plt.subplots(figsize=(14, 10))

# # Show ALL hives sorted by breakpoints
# breakpoint_summary_sorted = breakpoint_summary.sort_values(
#     'total_breakpoints', ascending=True
# )

# all_hives_sorted = breakpoint_summary_sorted

# hives = all_hives_sorted[HIVE_COLUMN].values
# breakpoints = all_hives_sorted['total_breakpoints'].values
# swarming = all_hives_sorted['swarming_events'].values

# colors = []
# for s in swarming:
#     if s >= 3:
#         colors.append('#E74C3C')  # Red - high swarming
#     elif s >= 1:
#         colors.append('#F39C12')  # Orange - swarming occurred
#     else:
#         colors.append('#3498DB')  # Blue - no swarming

# bars = ax.barh(hives, breakpoints, color=colors)

# for bar, value, s in zip(bars, breakpoints, swarming):
#     label = str(int(value))
#     if s > 0:
#         label += f" ({s} swarming events)"
#     ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
#             label, ha='left', va='center', fontsize=7)

# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='#E74C3C', label='3+ Swarming Events'),
#     Patch(facecolor='#F39C12', label='1-2 Swarming Events'),
#     Patch(facecolor='#3498DB', label='No Swarming Events')
# ]
# ax.legend(handles=legend_elements, loc='lower right')

# ax.set_xlabel('Total Breakpoints', fontsize=12)
# ax.set_ylabel('Hive ID', fontsize=12)
# ax.set_title(f'All {len(hives)} Hives: Breakpoints with Swarming Events', fontsize=14, fontweight='bold')
# ax.grid(axis='x', alpha=0.3)

# plt.tight_layout()
# SAVE_PATH = os.path.join(GRAPH_FOLDER, "pelt_breakpoints_barchart.png")
# plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"  Saved: {SAVE_PATH}")

# # -----------------------------------------------------
# # 3. REGIME PIE CHART
# # -----------------------------------------------------

# print("\n3. Creating Regime Distribution Charts...")

# def classify_regime(density):
#     if density == 0:
#         return 'Normal'
#     elif density <= 2:
#         return 'Changing'
#     else:
#         return 'Abnormal'

# df['regime'] = df['breakpoint_density'].apply(classify_regime)

# regime_counts = df['regime'].value_counts()
# regime_percent = df['regime'].value_counts(normalize=True) * 100

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# labels = ['Normal', 'Changing', 'Abnormal']
# values = [regime_counts.get('Normal', 0), 
#           regime_counts.get('Changing', 0),
#           regime_counts.get('Abnormal', 0)]
# colors = ['#27AE60', '#F39C12', '#E74C3C']

# wedges, texts, autotexts = ax1.pie(
#     values, 
#     labels=labels,
#     colors=colors,
#     autopct='%1.1f%%',
#     startangle=90,
#     explode=(0.02, 0.02, 0.05)
# )

# for text in texts:
#     text.set_fontsize(12)
# for autotext in autotexts:
#     autotext.set_color('white')
#     autotext.set_fontsize(14)
#     autotext.set_fontweight('bold')

# ax1.set_title('Regime Distribution', fontsize=14, fontweight='bold')

# bars = ax2.bar(labels, values, color=colors)

# for bar, value in zip(bars, values):
#     ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
#              f'{value:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# ax2.set_xlabel('Regime', fontsize=12)
# ax2.set_ylabel('Count', fontsize=12)
# ax2.set_title('Regime Count Distribution', fontsize=14, fontweight='bold')
# ax2.grid(axis='y', alpha=0.3)

# plt.tight_layout()
# SAVE_PATH = os.path.join(GRAPH_FOLDER, "pelt_regime_distribution.png")
# plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"  Saved: {SAVE_PATH}")

# # -----------------------------------------------------
# # 4. OVERALL STATISTICS TABLE
# # -----------------------------------------------------

# print("\n4. Creating Overall Statistics Table...")

# total_breakpoints = df['breakpoint'].sum()
# total_records = len(df)
# breakpoint_percentage = (total_breakpoints / total_records) * 100
# avg_density = df['breakpoint_density'].mean()
# max_density = df['breakpoint_density'].max()
# avg_segment = df['segment_duration'].mean()
# total_swarming = breakpoint_summary['swarming_events'].sum()
# hives_with_swarming = len(breakpoint_summary[breakpoint_summary['swarming_events'] > 0])

# fig, ax = plt.subplots(figsize=(8, 5))
# ax.axis('off')

# stats_data = [
#     ['Total Records', f'{total_records:,}'],
#     ['Total Breakpoints', f'{total_breakpoints:,}'],
#     ['Breakpoint Percentage', f'{breakpoint_percentage:.2f}%'],
#     ['Avg Breakpoint Density (24h)', f'{avg_density:.2f}'],
#     ['Max Breakpoint Density (24h)', f'{max_density:.0f}'],
#     ['Avg Segment Duration (hours)', f'{avg_segment:.2f}'],
#     ['Hives Analyzed', f'{df[HIVE_COLUMN].nunique():,}'],
#     ['Total Swarming Events (Unique Dates)', f'{total_swarming:,}'],
#     ['Hives with Swarming', f'{hives_with_swarming:,}']
# ]

# table = ax.table(
#     cellText=stats_data,
#     colLabels=['Metric', 'Value'],
#     cellLoc='center',
#     loc='center',
#     colWidths=[0.5, 0.4]
# )

# table.auto_set_font_size(False)
# table.set_fontsize(11)
# table.scale(1.2, 1.8)

# for j in range(2):
#     table[(0, j)].set_facecolor('#2C3E50')
#     table[(0, j)].set_text_props(color='white', fontweight='bold')

# for i in range(1, len(stats_data) + 1):
#     for j in range(2):
#         if i % 2 == 0:
#             table[(i, j)].set_facecolor('#ECF0F1')
#         else:
#             table[(i, j)].set_facecolor('#FFFFFF')

# ax.set_title('Table 2: Overall PELT Statistics with Swarming Events', 
#              fontsize=14, fontweight='bold', pad=20)

# plt.tight_layout()
# SAVE_PATH = os.path.join(GRAPH_FOLDER, "pelt_overall_statistics.png")
# plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', facecolor='white')
# plt.close()
# print(f"  Saved: {SAVE_PATH}")

# # -----------------------------------------------------
# # 5. SWARMING VS BREAKPOINTS SCATTER PLOT
# # -----------------------------------------------------

# print("\n5. Creating Swarming vs Breakpoints Scatter Plot...")

# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot swarming hives
# hives_with_swarming = breakpoint_summary[breakpoint_summary['swarming_events'] > 0]
# hives_no_swarming = breakpoint_summary[breakpoint_summary['swarming_events'] == 0]

# # Plot no-swarming hives (background)
# if len(hives_no_swarming) > 0:
#     ax.scatter(
#         hives_no_swarming['total_breakpoints'],
#         hives_no_swarming['swarming_events'] + np.random.uniform(-0.02, 0.02, len(hives_no_swarming)),
#         c='lightgray',
#         s=50,
#         alpha=0.5,
#         label='No Swarming'
#     )

# # Plot swarming hives (highlighted)
# if len(hives_with_swarming) > 0:
#     scatter = ax.scatter(
#         hives_with_swarming['total_breakpoints'],
#         hives_with_swarming['swarming_events'],
#         c=hives_with_swarming['breakpoints_per_100h'],
#         cmap='YlOrRd',
#         s=150,
#         alpha=0.8,
#         edgecolors='black',
#         linewidth=1,
#         label='Swarming Events'
#     )
#     cbar = plt.colorbar(scatter, ax=ax)
#     cbar.set_label('Breakpoints per 100h', fontsize=11)

# ax.set_xlabel('Total Breakpoints', fontsize=12)
# ax.set_ylabel('Swarming Events (Unique Dates)', fontsize=12)
# ax.set_title('Swarming Events vs Breakpoints Detected', fontsize=14, fontweight='bold')
# ax.grid(alpha=0.3)

# # Add correlation info
# if len(hives_with_swarming) > 0:
#     correlation = breakpoint_summary['total_breakpoints'].corr(breakpoint_summary['swarming_events'])
#     ax.text(0.95, 0.95, f'Correlation: {correlation:.3f}',
#             transform=ax.transAxes, ha='right', va='top',
#             fontsize=12, fontweight='bold',
#             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
# else:
#     ax.text(0.5, 0.5, 'No Swarming Events Found!',
#             transform=ax.transAxes, ha='center', va='center',
#             fontsize=16, fontweight='bold', color='red')

# plt.tight_layout()
# SAVE_PATH = os.path.join(GRAPH_FOLDER, "pelt_swarming_vs_breakpoints.png")
# plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"  Saved: {SAVE_PATH}")

# # -----------------------------------------------------
# # 6. CLEAN HEATMAP (No Overlapping Text)
# # -----------------------------------------------------

# print("\n6. Creating Clean Heatmap...")

# # Sample data for heatmap
# hive_sample = df[HIVE_COLUMN].unique()[:15]  # First 15 hives
# density_data = []
# hive_labels = []

# for hive in hive_sample:
#     hive_data = df[df[HIVE_COLUMN] == hive]
#     sample = hive_data['breakpoint_density'].iloc[::50].values
    
#     if len(sample) > 0:
#         max_len = 80
#         if len(sample) < max_len:
#             sample = np.pad(sample, (0, max_len - len(sample)), 'constant')
#         else:
#             sample = sample[:max_len]
#         density_data.append(sample)
#         hive_labels.append(hive)

# density_matrix = np.array(density_data)

# fig, ax = plt.subplots(figsize=(14, 8))

# im = ax.imshow(density_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

# ax.set_xlabel('Time (sampled)', fontsize=12)
# ax.set_ylabel('Hive ID', fontsize=12)
# ax.set_title('Breakpoint Density Heatmap (24-hour window)', fontsize=14, fontweight='bold')

# ax.set_yticks(range(len(hive_labels)))
# ax.set_yticklabels(hive_labels, fontsize=9)
# ax.set_xticks([])

# cbar = plt.colorbar(im, ax=ax)
# cbar.set_label('Breakpoint Density', fontsize=11)

# plt.tight_layout()
# SAVE_PATH = os.path.join(GRAPH_FOLDER, "pelt_heatmap_clean.png")
# plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
# plt.close()
# print(f"  Saved: {SAVE_PATH}")

# # -----------------------------------------------------
# # SAVE CSV SUMMARY
# # -----------------------------------------------------

# print("\nSaving CSV Summary...")

# summary_file = os.path.join(OUTPUT_FOLDER, "pelt_summary_with_swarming.csv")
# breakpoint_summary.to_csv(summary_file, index=False)
# print(f"  Saved: {summary_file}")

# # -----------------------------------------------------
# # SUMMARY
# # -----------------------------------------------------

# print("\n" + "=" * 60)
# print("ALL IMAGES GENERATED SUCCESSFULLY")
# print("=" * 60)

# print(f"\nTotal Swarming Events (Unique Dates): {total_swarming:,}")
# print(f"Hives with Swarming: {hives_with_swarming}")

# if total_swarming == 0:
#     print("\n" + "!" * 60)
#     print("WARNING: No swarming events found in the dataset!")
#     print(f"The '{TARGET_COLUMN}' column has all zeros.")
#     print("PELT breakpoints cannot predict swarming without swarming events.")
#     print("!" * 60)
# else:
#     print(f"\n✓ Swarming Events Found (Unique Dates): {total_swarming}")
#     print("✓ PELT + Swarming labels can now be used for prediction!")

# print("\nImages saved in:")
# print(f"  {GRAPH_FOLDER}")

# print("\nGenerated Images:")
# print("  1. pelt_breakpoint_summary.png - ALL hives with Swarming Events")
# print("  2. pelt_breakpoints_barchart.png - ALL hives with Swarming indicators")
# print("  3. pelt_regime_distribution.png - Regime pie + bar chart")
# print("  4. pelt_overall_statistics.png - Overall statistics with Swarming")
# print("  5. pelt_swarming_vs_breakpoints.png - Scatter plot with correlation")
# print("  6. pelt_heatmap_clean.png - Clean heatmap (NO overlapping text)")

# print("\nCSV File:")
# print(f"  - pelt_summary_with_swarming.csv")

# print("\n" + "=" * 60)
# print("COMPLETED")
# print("=" * 60)
 
"""
=========================================================
PELT RESULTS - CLEAN IMAGES FOR PUBLICATION (UPDATED)
=========================================================
(Auto-deletes existing images before generating new ones)
(Includes all new per-variable and alignment PELT features)
(Includes Breakpoint Timeline Visualization)
(Includes Breakpoint Alignment Table with 6h/12h/24h/48h windows for ALL 48 hives)
(CORRECTED: Swarming events counted as UNIQUE DATES, not hours)
(Includes Aligned Breakpoint Dates Table)
=========================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import *

print("=" * 60)
print("GENERATING CLEAN PELT RESULTS IMAGES (UPDATED)")
print("=" * 60)

# -----------------------------------------------------
# Create folders
# -----------------------------------------------------

GRAPH_FOLDER = os.path.join(OUTPUT_FOLDER, "graphs")
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# -----------------------------------------------------
# DELETE EXISTING PELT IMAGES
# -----------------------------------------------------

print("\n" + "=" * 60)
print("DELETING EXISTING PELT IMAGES")
print("=" * 60)

image_patterns = [
    "pelt_breakpoint_summary.png",
    "pelt_breakpoints_barchart.png",
    "pelt_regime_distribution.png",
    "pelt_overall_statistics.png",
    "pelt_heatmap_clean.png",
    "pelt_swarming_vs_breakpoints.png",
    "pelt_summary_with_swarming.png",
    "pelt_per_variable_breakpoints.png",
    "pelt_alignment_features.png",
    "breakpoint_timeline_*.png",
    "per_variable_breakpoints_*.png",
    "breakpoint_alignment_table.png",
    "breakpoint_alignment_24h_table.png",
    "breakpoint_alignment_24h_all_hives.png",
    "aligned_breakpoint_dates.png",  # NEW
]

deleted_count = 0
for pattern in image_patterns:
    file_path = os.path.join(GRAPH_FOLDER, pattern)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"  Deleted: {pattern}")
        deleted_count += 1

if deleted_count == 0:
    print("  No existing images found to delete")
else:
    print(f"\n  Total deleted: {deleted_count} images")

print("\n" + "=" * 60)
print("GENERATING NEW IMAGES")
print("=" * 60)

# -----------------------------------------------------
# Load data
# -----------------------------------------------------

print("\nLoading data...")

DATA_FILE = os.path.join(OUTPUT_FOLDER, "hive_data_with_pelt.csv")
df = pd.read_csv(DATA_FILE)

print(f"  Loaded: {DATA_FILE}")
print(f"  Records: {len(df):,}")
print(f"  Hives: {df[HIVE_COLUMN].nunique():,}")

# Print new PELT features if they exist
new_features = []
for col in df.columns:
    if col.startswith(('breakpoint_', 'days_since_breakpoint_', 'breakpoint_density_', 
                       'alignment_', 'all_aligned', 'majority_aligned', 'aligned_')):
        new_features.append(col)

if new_features:
    print(f"\n  Found {len(new_features)} new PELT features")
    print(f"  Example: {new_features[:5]}...")

# -----------------------------------------------------
# Convert timestamp to date for unique event counting
# -----------------------------------------------------

print("\nConverting timestamps to dates...")

df['date'] = pd.to_datetime(df[TIMESTAMP_COLUMN]).dt.date

print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Unique dates: {df['date'].nunique():,}")

# -----------------------------------------------------
# Check for swarming labels
# -----------------------------------------------------

print("\n" + "=" * 60)
print("SWARMING EVENTS ANALYSIS")
print("=" * 60)

SWARMING_LABEL = TARGET_COLUMN  # "swarming_label_next_72h"
SWARMING_EVENT_COLUMN = "swarming_event_label"  # Column W

# Check which columns exist
if SWARMING_EVENT_COLUMN in df.columns:
    print(f"\n1. Using '{SWARMING_EVENT_COLUMN}' for event counting...")
    
    swarming_dates = df[df[SWARMING_EVENT_COLUMN] == 1]['date'].unique()
    total_events = len(swarming_dates)
    
    print(f"   Total UNIQUE swarming dates: {total_events:,}")
    print(f"   Total rows with label=1: {df[SWARMING_EVENT_COLUMN].sum():,}")
    print(f"   Average hours per event: {df[SWARMING_EVENT_COLUMN].sum() / total_events if total_events > 0 else 0:.1f}")
    
    events_per_hive = df[df[SWARMING_EVENT_COLUMN] == 1].groupby(HIVE_COLUMN)['date'].nunique().reset_index()
    events_per_hive.columns = [HIVE_COLUMN, 'swarming_events']
    
    print(f"\n   Hives with events: {len(events_per_hive)}")
    print(f"   Total unique events: {events_per_hive['swarming_events'].sum():,}")
    
elif SWARMING_LABEL in df.columns:
    print(f"\n1. '{SWARMING_EVENT_COLUMN}' not found. Using '{SWARMING_LABEL}'...")
    
    swarming_dates = df[df[SWARMING_LABEL] == 1]['date'].unique()
    total_events = len(swarming_dates)
    
    print(f"   Total UNIQUE swarming dates: {total_events:,}")
    print(f"   Total rows with label=1: {df[SWARMING_LABEL].sum():,}")
    
    events_per_hive = df[df[SWARMING_LABEL] == 1].groupby(HIVE_COLUMN)['date'].nunique().reset_index()
    events_per_hive.columns = [HIVE_COLUMN, 'swarming_events']
    
else:
    print("\n  ERROR: No swarming labels found!")
    events_per_hive = pd.DataFrame({HIVE_COLUMN: [], 'swarming_events': []})

# -----------------------------------------------------
# Create Breakpoint Summary with Swarming Events
# -----------------------------------------------------

print("\n" + "=" * 60)
print("CREATING BREAKPOINT SUMMARY")
print("=" * 60)

breakpoint_summary = df.groupby(HIVE_COLUMN).agg(
    total_records=('breakpoint', 'count'),
    total_breakpoints=('breakpoint', 'sum'),
    breakpoint_density_avg=('breakpoint_density', 'mean'),
    max_breakpoint_density=('breakpoint_density', 'max')
).reset_index()

breakpoint_summary['breakpoints_per_100h'] = (
    breakpoint_summary['total_breakpoints'] / 
    breakpoint_summary['total_records'] * 100
)

# Merge with swarming events (unique counts)
if len(events_per_hive) > 0:
    breakpoint_summary = breakpoint_summary.merge(
        events_per_hive, 
        on=HIVE_COLUMN, 
        how='left'
    )
    breakpoint_summary['swarming_events'] = breakpoint_summary['swarming_events'].fillna(0).astype(int)
else:
    breakpoint_summary['swarming_events'] = 0

# Sort by breakpoints descending
breakpoint_summary = breakpoint_summary.sort_values(
    'total_breakpoints', ascending=False
)

total_events = breakpoint_summary['swarming_events'].sum()
hives_with_events = len(breakpoint_summary[breakpoint_summary['swarming_events'] > 0])

print(f"\nBreakpoint Summary Created:")
print(f"  Total Hives: {len(breakpoint_summary)}")
print(f"  Total Actual Swarming Events: {total_events:,}")
print(f"  Hives with Swarming: {hives_with_events}")

# =========================================================
# 1. BREAKPOINT SUMMARY TABLE - ALL HIVES
# =========================================================

print("\n1. Creating Breakpoint Summary Table with ALL Hives...")

all_hives = breakpoint_summary
num_hives = len(all_hives)
fig_height = max(8, num_hives * 0.32)

fig, ax = plt.subplots(figsize=(14, fig_height))
ax.axis('off')

table_data = all_hives.values.tolist()
col_labels = ['Hive', 'Records', 'Breakpoints', 'Avg Density', 'Max Density', 'per 100h', 'Swarming Events']

formatted_data = []
for row in table_data:
    formatted_data.append([
        row[0],
        f"{row[1]:,}",
        f"{row[2]:,}",
        f"{row[3]:.2f}",
        f"{row[4]:.0f}",
        f"{row[5]:.2f}",
        f"{row[6]:,}" if len(row) > 6 else "0"
    ])

font_size = max(6, min(9, 14 - num_hives * 0.06))

table = ax.table(
    cellText=formatted_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    colWidths=[0.10, 0.12, 0.14, 0.14, 0.12, 0.14, 0.14]
)

table.auto_set_font_size(False)
table.set_fontsize(font_size)
table.scale(1.2, 1.5)

for j in range(len(col_labels)):
    table[(0, j)].set_facecolor('#2C3E50')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

for i in range(1, len(formatted_data) + 1):
    row_data = table_data[i-1]
    for j in range(len(col_labels)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ECF0F1')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')
    
    if len(row_data) > 6 and row_data[6] > 0:
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor('#FFF3E0')
            if j == len(col_labels) - 1:
                if row_data[6] >= 3:
                    table[(i, j)].set_facecolor('#E74C3C')
                elif row_data[6] >= 2:
                    table[(i, j)].set_facecolor('#F39C12')
                else:
                    table[(i, j)].set_facecolor('#3498DB')
                table[(i, j)].set_text_props(color='white', fontweight='bold')

ax.set_title(f'Table 1: All {num_hives} Hives with Breakpoints and Swarming Events', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
SAVE_PATH = os.path.join(GRAPH_FOLDER, "pelt_breakpoint_summary.png")
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {SAVE_PATH} (All {num_hives} hives)")

# =========================================================
# 2. BREAKPOINTS BAR CHART WITH SWARMING INDICATORS
# =========================================================

print("\n2. Creating Breakpoint Bar Chart with Swarming Indicators...")

fig, ax = plt.subplots(figsize=(14, 10))

breakpoint_summary_sorted = breakpoint_summary.sort_values(
    'total_breakpoints', ascending=True
)

all_hives_sorted = breakpoint_summary_sorted

hives = all_hives_sorted[HIVE_COLUMN].values
breakpoints = all_hives_sorted['total_breakpoints'].values
swarming = all_hives_sorted['swarming_events'].values

colors = []
for s in swarming:
    if s >= 3:
        colors.append('#E74C3C')
    elif s >= 1:
        colors.append('#F39C12')
    else:
        colors.append('#3498DB')

bars = ax.barh(hives, breakpoints, color=colors)

for bar, value, s in zip(bars, breakpoints, swarming):
    label = str(int(value))
    if s > 0:
        label += f" ({s} swarming events)"
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            label, ha='left', va='center', fontsize=7)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E74C3C', label='3+ Swarming Events'),
    Patch(facecolor='#F39C12', label='1-2 Swarming Events'),
    Patch(facecolor='#3498DB', label='No Swarming Events')
]
ax.legend(handles=legend_elements, loc='lower right')

ax.set_xlabel('Total Breakpoints', fontsize=12)
ax.set_ylabel('Hive ID', fontsize=12)
ax.set_title(f'All {len(hives)} Hives: Breakpoints with Swarming Events', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
SAVE_PATH = os.path.join(GRAPH_FOLDER, "pelt_breakpoints_barchart.png")
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {SAVE_PATH}")

# =========================================================
# 3. REGIME PIE CHART
# =========================================================

print("\n3. Creating Regime Distribution Charts...")

def classify_regime(density):
    if density == 0:
        return 'Normal'
    elif density <= 2:
        return 'Changing'
    else:
        return 'Abnormal'

df['regime'] = df['breakpoint_density'].apply(classify_regime)

regime_counts = df['regime'].value_counts()
regime_percent = df['regime'].value_counts(normalize=True) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

labels = ['Normal', 'Changing', 'Abnormal']
values = [regime_counts.get('Normal', 0), 
          regime_counts.get('Changing', 0),
          regime_counts.get('Abnormal', 0)]
colors = ['#27AE60', '#F39C12', '#E74C3C']

wedges, texts, autotexts = ax1.pie(
    values, 
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    explode=(0.02, 0.02, 0.05)
)

for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')

ax1.set_title('Regime Distribution', fontsize=14, fontweight='bold')

bars = ax2.bar(labels, values, color=colors)

for bar, value in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'{value:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_xlabel('Regime', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Regime Count Distribution', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
SAVE_PATH = os.path.join(GRAPH_FOLDER, "pelt_regime_distribution.png")
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {SAVE_PATH}")

# =========================================================
# 4. OVERALL STATISTICS TABLE
# =========================================================

print("\n4. Creating Overall Statistics Table...")

total_breakpoints = df['breakpoint'].sum()
total_records = len(df)
breakpoint_percentage = (total_breakpoints / total_records) * 100
avg_density = df['breakpoint_density'].mean()
max_density = df['breakpoint_density'].max()
avg_segment = df['segment_duration'].mean()
total_swarming = breakpoint_summary['swarming_events'].sum()
hives_with_swarming = len(breakpoint_summary[breakpoint_summary['swarming_events'] > 0])

fig, ax = plt.subplots(figsize=(8, 5))
ax.axis('off')

stats_data = [
    ['Total Records', f'{total_records:,}'],
    ['Total Breakpoints', f'{total_breakpoints:,}'],
    ['Breakpoint Percentage', f'{breakpoint_percentage:.2f}%'],
    ['Avg Breakpoint Density (24h)', f'{avg_density:.2f}'],
    ['Max Breakpoint Density (24h)', f'{max_density:.0f}'],
    ['Avg Segment Duration (hours)', f'{avg_segment:.2f}'],
    ['Hives Analyzed', f'{df[HIVE_COLUMN].nunique():,}'],
    ['Total Swarming Events (Unique Dates)', f'{total_swarming:,}'],
    ['Hives with Swarming', f'{hives_with_swarming:,}']
]

table = ax.table(
    cellText=stats_data,
    colLabels=['Metric', 'Value'],
    cellLoc='center',
    loc='center',
    colWidths=[0.5, 0.4]
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

for j in range(2):
    table[(0, j)].set_facecolor('#2C3E50')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

for i in range(1, len(stats_data) + 1):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ECF0F1')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

ax.set_title('Table 2: Overall PELT Statistics with Swarming Events', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
SAVE_PATH = os.path.join(GRAPH_FOLDER, "pelt_overall_statistics.png")
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {SAVE_PATH}")

# =========================================================
# 5. SWARMING VS BREAKPOINTS SCATTER PLOT
# =========================================================

print("\n5. Creating Swarming vs Breakpoints Scatter Plot...")

fig, ax = plt.subplots(figsize=(10, 6))

hives_with_swarming = breakpoint_summary[breakpoint_summary['swarming_events'] > 0]
hives_no_swarming = breakpoint_summary[breakpoint_summary['swarming_events'] == 0]

if len(hives_no_swarming) > 0:
    ax.scatter(
        hives_no_swarming['total_breakpoints'],
        hives_no_swarming['swarming_events'] + np.random.uniform(-0.02, 0.02, len(hives_no_swarming)),
        c='lightgray',
        s=50,
        alpha=0.5,
        label='No Swarming'
    )

if len(hives_with_swarming) > 0:
    scatter = ax.scatter(
        hives_with_swarming['total_breakpoints'],
        hives_with_swarming['swarming_events'],
        c=hives_with_swarming['breakpoints_per_100h'],
        cmap='YlOrRd',
        s=150,
        alpha=0.8,
        edgecolors='black',
        linewidth=1,
        label='Swarming Events'
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Breakpoints per 100h', fontsize=11)

ax.set_xlabel('Total Breakpoints', fontsize=12)
ax.set_ylabel('Swarming Events (Unique Dates)', fontsize=12)
ax.set_title('Swarming Events vs Breakpoints Detected', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

if len(hives_with_swarming) > 0:
    correlation = breakpoint_summary['total_breakpoints'].corr(breakpoint_summary['swarming_events'])
    ax.text(0.95, 0.95, f'Correlation: {correlation:.3f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
else:
    ax.text(0.5, 0.5, 'No Swarming Events Found!',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=16, fontweight='bold', color='red')

plt.tight_layout()
SAVE_PATH = os.path.join(GRAPH_FOLDER, "pelt_swarming_vs_breakpoints.png")
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {SAVE_PATH}")

# =========================================================
# 6. CLEAN HEATMAP
# =========================================================

print("\n6. Creating Clean Heatmap...")

hive_sample = df[HIVE_COLUMN].unique()[:15]
density_data = []
hive_labels = []

for hive in hive_sample:
    hive_data = df[df[HIVE_COLUMN] == hive]
    sample = hive_data['breakpoint_density'].iloc[::50].values
    
    if len(sample) > 0:
        max_len = 80
        if len(sample) < max_len:
            sample = np.pad(sample, (0, max_len - len(sample)), 'constant')
        else:
            sample = sample[:max_len]
        density_data.append(sample)
        hive_labels.append(hive)

density_matrix = np.array(density_data)

fig, ax = plt.subplots(figsize=(14, 8))

im = ax.imshow(density_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

ax.set_xlabel('Time (sampled)', fontsize=12)
ax.set_ylabel('Hive ID', fontsize=12)
ax.set_title('Breakpoint Density Heatmap (24-hour window)', fontsize=14, fontweight='bold')

ax.set_yticks(range(len(hive_labels)))
ax.set_yticklabels(hive_labels, fontsize=9)
ax.set_xticks([])

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Breakpoint Density', fontsize=11)

plt.tight_layout()
SAVE_PATH = os.path.join(GRAPH_FOLDER, "pelt_heatmap_clean.png")
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {SAVE_PATH}")

# =========================================================
# 7. PER-VARIABLE BREAKPOINT COMPARISON
# =========================================================

print("\n7. Creating Per-Variable Breakpoint Comparison...")

var_names = ['temp', 'hum', 'co2', 'weight']
var_labels = ['Temperature', 'Humidity', 'CO₂', 'Weight']
var_colors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b']

var_breakpoints = []
for var in var_names:
    col = f'breakpoint_{var}'
    if col in df.columns:
        var_breakpoints.append(df[col].sum())
    else:
        var_breakpoints.append(0)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(var_labels, var_breakpoints, color=var_colors)

for bar, value in zip(bars, var_breakpoints):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(int(value)), ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xlabel('Variable', fontsize=12)
ax.set_ylabel('Number of Breakpoints', fontsize=12)
ax.set_title('Breakpoints Detected per Variable', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
SAVE_PATH = os.path.join(GRAPH_FOLDER, "pelt_per_variable_breakpoints.png")
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {SAVE_PATH}")

# =========================================================
# 8. ALIGNMENT FEATURES
# =========================================================

print("\n8. Creating Alignment Features Charts...")

if 'alignment_count' in df.columns:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    alignment_counts = df['alignment_count'].value_counts().sort_index()
    
    bars = ax1.bar(alignment_counts.index, alignment_counts.values, 
                   color=['#3498DB', '#2ECC71', '#F39C12', '#E74C3C', '#9B59B6'])
    
    for bar, value in zip(bars, alignment_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'{value:,}', ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Number of Variables Aligned (0-4)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Alignment Count Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    if 'alignment_ratio' in df.columns:
        hive_alignment = df.groupby(HIVE_COLUMN)['alignment_ratio'].mean().reset_index()
        
        ax2.barh(hive_alignment[HIVE_COLUMN].head(15), 
                 hive_alignment['alignment_ratio'].head(15),
                 color='#3498DB')
        ax2.set_xlabel('Average Alignment Ratio (0-1)', fontsize=12)
        ax2.set_ylabel('Hive ID', fontsize=12)
        ax2.set_title('Top 15 Hives: Average Alignment Ratio', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    SAVE_PATH = os.path.join(GRAPH_FOLDER, "pelt_alignment_features.png")
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {SAVE_PATH}")

# =========================================================
# 9. BREAKPOINT TIMELINE VISUALIZATION
# =========================================================

print("\n9. Creating Breakpoint Timeline Visualizations...")

hives_with_swarms = breakpoint_summary[breakpoint_summary['swarming_events'] > 0][HIVE_COLUMN].unique()

if len(hives_with_swarms) > 0:
    hives_to_plot = hives_with_swarms[:min(3, len(hives_with_swarms))]
    print(f"   Plotting breakpoints for hives: {list(hives_to_plot)}")
    
    for hive_to_plot in hives_to_plot:
        print(f"   Processing hive: {hive_to_plot}")
        
        hive_data = df[df[HIVE_COLUMN] == hive_to_plot].copy()
        hive_data = hive_data.sort_values(TIMESTAMP_COLUMN)
        hive_data[TIMESTAMP_COLUMN] = pd.to_datetime(hive_data[TIMESTAMP_COLUMN])
        
        # 9a. Breakpoint Timeline - All Variables
        fig, ax = plt.subplots(figsize=(16, 8))
        
        variables = [
            {"col": "breakpoint_temp", "label": "Temperature", "color": "#ef4444", "y": 4},
            {"col": "breakpoint_hum", "label": "Humidity", "color": "#3b82f6", "y": 3},
            {"col": "breakpoint_co2", "label": "CO₂", "color": "#22c55e", "y": 2},
            {"col": "breakpoint_weight", "label": "Weight", "color": "#f59e0b", "y": 1},
        ]
        
        for var in variables:
            col = var["col"]
            if col in hive_data.columns:
                bp_times = hive_data[hive_data[col] == 1][TIMESTAMP_COLUMN].values
                bp_values = [var["y"]] * len(bp_times)
                ax.scatter(bp_times, bp_values, 
                           color=var["color"], 
                           s=80, 
                           marker="|",
                           linewidth=3,
                           label=var["label"],
                           zorder=5)
        
        if "breakpoint" in hive_data.columns:
            bp_times = hive_data[hive_data["breakpoint"] == 1][TIMESTAMP_COLUMN].values
            bp_values = [0] * len(bp_times)
            ax.scatter(bp_times, bp_values, 
                       color="#8b5cf6", 
                       s=150, 
                       marker="*",
                       linewidth=2,
                       label="Multivariate PELT (All 4)",
                       zorder=10)
        
        if "swarming_event_label" in hive_data.columns:
            swarm_times = hive_data[hive_data["swarming_event_label"] == 1][TIMESTAMP_COLUMN].values
            swarm_values = [0] * len(swarm_times)
            ax.scatter(swarm_times, swarm_values, 
                       color="#ff0000", 
                       s=200, 
                       marker="^",
                       linewidth=2,
                       label="Swarming Event",
                       zorder=10)
        
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["Multivariate\nPELT", "Weight", "CO₂", "Humidity", "Temperature"])
        ax.set_ylim(-0.5, 4.5)
        
        ax.set_xlabel("Date/Time", fontsize=12)
        ax.set_title(f"Breakpoints Over Time - Hive: {hive_to_plot}", fontsize=16, fontweight='bold')
        
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, ha='right')
        
        if "alignment_count" in hive_data.columns:
            aligned_3 = hive_data[hive_data["alignment_count"] >= 3][TIMESTAMP_COLUMN].values
            for t in aligned_3:
                ax.axvline(x=t, color='#fbbf24', alpha=0.15, linewidth=2)
            
            ax.text(0.02, 0.98, 
                    "Yellow lines = 3+ variables aligned",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        SAVE_PATH = os.path.join(GRAPH_FOLDER, f"breakpoint_timeline_{hive_to_plot}.png")
        plt.tight_layout()
        plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"     Saved: breakpoint_timeline_{hive_to_plot}.png")
        
        # 9b. Per-Variable Breakpoints - 4 Subplots
        fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
        
        var_colors = {
            "breakpoint_temp": ("#ef4444", "Temperature"),
            "breakpoint_hum": ("#3b82f6", "Humidity"),
            "breakpoint_co2": ("#22c55e", "CO₂"),
            "breakpoint_weight": ("#f59e0b", "Weight")
        }
        
        for idx, (col, (color, label)) in enumerate(var_colors.items()):
            ax = axes[idx]
            
            if col in hive_data.columns:
                bp_times = hive_data[hive_data[col] == 1][TIMESTAMP_COLUMN].values
                ax.scatter(bp_times, [1] * len(bp_times), 
                           color=color, s=50, marker="|", linewidth=2)
                
                if "swarming_event_label" in hive_data.columns:
                    swarm_times = hive_data[hive_data["swarming_event_label"] == 1][TIMESTAMP_COLUMN].values
                    ax.scatter(swarm_times, [1] * len(swarm_times), 
                               color="#ff0000", s=100, marker="^", 
                               label="Swarming Event")
                
                ax.set_ylabel(label, fontsize=11)
                ax.set_ylim(0.5, 1.5)
                ax.set_yticks([1])
                ax.set_yticklabels(["Breakpoint"])
                ax.grid(True, alpha=0.2)
                
                if idx == 3:
                    ax.set_xlabel("Date/Time", fontsize=12)
                
                if idx == 0:
                    ax.legend(loc="upper right", fontsize=9)
        
        fig.suptitle(f"Per-Variable Breakpoints and Swarming Events - Hive: {hive_to_plot}", 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        SAVE_PATH = os.path.join(GRAPH_FOLDER, f"per_variable_breakpoints_{hive_to_plot}.png")
        plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"     Saved: per_variable_breakpoints_{hive_to_plot}.png")

else:
    print("   No hives with swarming events found. Skipping timeline visualization.")

# =========================================================
# 10. BREAKPOINT ALIGNMENT TABLE - ALL 48 HIVES
# =========================================================

print("\n10. Creating Breakpoint Alignment Table for ALL 48 Hives (24-hour window)...")

# Function to get breakpoint timestamps for a hive
def get_breakpoint_times(hive_data, var_name):
    col = f'breakpoint_{var_name}'
    if col in hive_data.columns:
        times = hive_data[hive_data[col] == 1][TIMESTAMP_COLUMN].values
        return times
    return []

# Function to check alignment within a time window
def check_alignment_window(times1, times2, times3, times4, window_hours=24):
    """
    Check if all 4 variables have breakpoints within the same time window.
    Returns the number of unique windows where all 4 variables aligned.
    """
    if len(times1) == 0 or len(times2) == 0 or len(times3) == 0 or len(times4) == 0:
        return 0
    
    t1 = pd.to_datetime(times1)
    t2 = pd.to_datetime(times2)
    t3 = pd.to_datetime(times3)
    t4 = pd.to_datetime(times4)
    
    aligned_windows = []
    half_window = window_hours / 2
    
    for time1 in t1:
        window_start = time1 - pd.Timedelta(hours=half_window)
        window_end = time1 + pd.Timedelta(hours=half_window)
        
        has_t2 = any((t2 >= window_start) & (t2 <= window_end))
        has_t3 = any((t3 >= window_start) & (t3 <= window_end))
        has_t4 = any((t4 >= window_start) & (t4 <= window_end))
        
        if has_t2 and has_t3 and has_t4:
            aligned_windows.append(time1.strftime('%Y-%m-%d'))
    
    return len(set(aligned_windows))

# Get ALL 48 hives sorted by swarming events
all_hives_sorted = breakpoint_summary.sort_values('swarming_events', ascending=False)[HIVE_COLUMN].values

print(f"   Creating alignment table for ALL {len(all_hives_sorted)} hives...")

# Prepare table data for ALL hives
table_data_all = []
for hive in all_hives_sorted:
    hive_data = df[df[HIVE_COLUMN] == hive].copy()
    hive_data = hive_data.sort_values(TIMESTAMP_COLUMN)
    hive_data[TIMESTAMP_COLUMN] = pd.to_datetime(hive_data[TIMESTAMP_COLUMN])
    
    co2_times = get_breakpoint_times(hive_data, 'co2')
    temp_times = get_breakpoint_times(hive_data, 'temp')
    hum_times = get_breakpoint_times(hive_data, 'hum')
    weight_times = get_breakpoint_times(hive_data, 'weight')
    
    swarm_count = 0
    if 'swarming_event_label' in hive_data.columns:
        swarm_times = hive_data[hive_data['swarming_event_label'] == 1][TIMESTAMP_COLUMN].values
        if len(swarm_times) > 0:
            swarm_dates = set([pd.to_datetime(t).date() for t in swarm_times])
            swarm_count = len(swarm_dates)
    
    aligned_6h = check_alignment_window(co2_times, temp_times, hum_times, weight_times, window_hours=6)
    aligned_12h = check_alignment_window(co2_times, temp_times, hum_times, weight_times, window_hours=12)
    aligned_24h = check_alignment_window(co2_times, temp_times, hum_times, weight_times, window_hours=24)
    aligned_48h = check_alignment_window(co2_times, temp_times, hum_times, weight_times, window_hours=48)
    
    table_data_all.append({
        'Hive': hive,
        'CO₂ BP': len(co2_times),
        'Temp BP': len(temp_times),
        'Humidity BP': len(hum_times),
        'Weight BP': len(weight_times),
        'Aligned 6h': aligned_6h,
        'Aligned 12h': aligned_12h,
        'Aligned 24h': aligned_24h,
        'Aligned 48h': aligned_48h,
        'Swarming': swarm_count
    })

alignment_df_all = pd.DataFrame(table_data_all)
alignment_df_all = alignment_df_all.sort_values('Aligned 24h', ascending=False)

print("\n   Breakpoint Alignment Summary (24-hour window) - ALL 48 Hives:")
print(alignment_df_all.to_string(index=False))

csv_path_all = os.path.join(OUTPUT_FOLDER, "breakpoint_alignment_24h_all_hives.csv")
alignment_df_all.to_csv(csv_path_all, index=False)
print(f"\n   CSV saved: {csv_path_all}")

# Create table image for ALL 48 hives
print("\n   Creating alignment table image for ALL 48 hives...")

num_hives_all = len(alignment_df_all)
fig_height = max(8, num_hives_all * 0.35)

fig, ax = plt.subplots(figsize=(18, fig_height))
ax.axis('off')

col_labels = ['Hive', 'CO₂ BP', 'Temp BP', 'Humidity BP', 'Weight BP', 
              'Aligned 6h', 'Aligned 12h', 'Aligned 24h', 'Aligned 48h', 'Swarming (Unique Dates)']
table_data_rows = alignment_df_all.values.tolist()

formatted_rows = []
for row in table_data_rows:
    formatted_rows.append([
        row[0],
        str(int(row[1])),
        str(int(row[2])),
        str(int(row[3])),
        str(int(row[4])),
        str(int(row[5])),
        str(int(row[6])),
        str(int(row[7])),
        str(int(row[8])),
        str(int(row[9]))
    ])

font_size = max(6, min(9, 12 - num_hives_all * 0.04))

table = ax.table(
    cellText=formatted_rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    colWidths=[0.08, 0.07, 0.07, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09]
)

table.auto_set_font_size(False)
table.set_fontsize(font_size)
table.scale(1.2, 1.5)

for j in range(len(col_labels)):
    table[(0, j)].set_facecolor('#2C3E50')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

for i in range(1, len(formatted_rows) + 1):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ECF0F1')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')
    
    if int(formatted_rows[i-1][7]) > 0:
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor('#FFF3E0')
            if j >= 5 and j <= 8:
                if int(formatted_rows[i-1][j]) >= 5:
                    table[(i, j)].set_facecolor('#E74C3C')
                    table[(i, j)].set_text_props(color='white', fontweight='bold')
                elif int(formatted_rows[i-1][j]) >= 2:
                    table[(i, j)].set_facecolor('#F39C12')
                    table[(i, j)].set_text_props(color='white', fontweight='bold')
                else:
                    table[(i, j)].set_facecolor('#3498DB')
                    table[(i, j)].set_text_props(color='white', fontweight='bold')

ax.set_title(f'Table 3: Breakpoint Alignment Summary - ALL {num_hives_all} Hives (Multiple Time Windows)', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
SAVE_PATH = os.path.join(GRAPH_FOLDER, "breakpoint_alignment_24h_all_hives.png")
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   Table saved: {SAVE_PATH}")

# =========================================================
# 10b. Summary Statistics for All Hives
# =========================================================

print("\n   Calculating alignment statistics for all hives...")

print("\n   Alignment Statistics Summary:")
print(f"   Total Hives with at least 1 alignment (24h): {(alignment_df_all['Aligned 24h'] > 0).sum()}")
print(f"   Total Hives with 2+ alignments (24h): {(alignment_df_all['Aligned 24h'] >= 2).sum()}")
print(f"   Total Hives with 5+ alignments (24h): {(alignment_df_all['Aligned 24h'] >= 5).sum()}")
print(f"   Max alignments in a single hive (24h): {alignment_df_all['Aligned 24h'].max()}")

if len(alignment_df_all) > 0:
    corr_24h = alignment_df_all['Aligned 24h'].corr(alignment_df_all['Swarming'])
    print(f"   Correlation (24h alignments vs Swarming - Unique Dates): {corr_24h:.4f}")

stats_path_all = os.path.join(OUTPUT_FOLDER, "alignment_statistics_all_hives.csv")
alignment_df_all.to_csv(stats_path_all, index=False)
print(f"   Alignment statistics saved: {stats_path_all}")

# =========================================================
# 10c. ALIGNED BREAKPOINT DATES TABLE (NEW) - FIXED
# =========================================================

print("\n10c. Creating Aligned Breakpoint Dates Table...")

def get_aligned_dates(hive_data, window_hours=24):
    """
    Find dates when all 4 variables have breakpoints within the same window.
    Returns a list of dates and the variables that aligned.
    """
    co2_times = get_breakpoint_times(hive_data, 'co2')
    temp_times = get_breakpoint_times(hive_data, 'temp')
    hum_times = get_breakpoint_times(hive_data, 'hum')
    weight_times = get_breakpoint_times(hive_data, 'weight')
    
    if len(co2_times) == 0 or len(temp_times) == 0 or len(hum_times) == 0 or len(weight_times) == 0:
        return []
    
    t1 = pd.to_datetime(co2_times)
    t2 = pd.to_datetime(temp_times)
    t3 = pd.to_datetime(hum_times)
    t4 = pd.to_datetime(weight_times)
    
    aligned_dates = []
    half_window = window_hours / 2
    
    for time1 in t1:
        window_start = time1 - pd.Timedelta(hours=half_window)
        window_end = time1 + pd.Timedelta(hours=half_window)
        
        has_t2 = any((t2 >= window_start) & (t2 <= window_end))
        has_t3 = any((t3 >= window_start) & (t3 <= window_end))
        has_t4 = any((t4 >= window_start) & (t4 <= window_end))
        
        if has_t2 and has_t3 and has_t4:
            aligned_dates.append({
                'date': time1.strftime('%Y-%m-%d'),
                'datetime': time1,
                'window': f'{window_start.strftime("%Y-%m-%d %H:%M")} - {window_end.strftime("%Y-%m-%d %H:%M")}'
            })
    
    seen = set()
    unique_dates = []
    for item in aligned_dates:
        if item['date'] not in seen:
            seen.add(item['date'])
            unique_dates.append(item)
    
    return unique_dates

# ✅ FIXED: Use 'Hive' instead of HIVE_COLUMN
top_aligned_hives = alignment_df_all.nlargest(5, 'Aligned 24h')['Hive'].values

if len(top_aligned_hives) > 0:
    print(f"   Finding aligned dates for top {len(top_aligned_hives)} hives: {list(top_aligned_hives)}")
    
    aligned_dates_data = []
    for hive in top_aligned_hives:
        hive_data = df[df[HIVE_COLUMN] == hive].copy()
        hive_data = hive_data.sort_values(TIMESTAMP_COLUMN)
        hive_data[TIMESTAMP_COLUMN] = pd.to_datetime(hive_data[TIMESTAMP_COLUMN])
        
        dates = get_aligned_dates(hive_data, window_hours=24)
        
        for item in dates:
            aligned_dates_data.append({
                'Hive': hive,
                'Date': item['date'],
                'Window': item['window']
            })
    
    if len(aligned_dates_data) > 0:
        aligned_dates_df = pd.DataFrame(aligned_dates_data)
        aligned_dates_df = aligned_dates_df.sort_values(['Hive', 'Date'])
        
        print("\n   Aligned Breakpoint Dates:")
        print(aligned_dates_df.to_string(index=False))
        
        dates_csv_path = os.path.join(OUTPUT_FOLDER, "aligned_breakpoint_dates.csv")
        aligned_dates_df.to_csv(dates_csv_path, index=False)
        print(f"\n   CSV saved: {dates_csv_path}")
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(aligned_dates_df) * 0.3)))
        ax.axis('off')
        
        col_labels = ['Hive', 'Date', 'Window (24h)']
        table_data_dates = aligned_dates_df.values.tolist()
        
        table_dates = ax.table(
            cellText=table_data_dates,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            colWidths=[0.15, 0.25, 0.50]
        )
        
        table_dates.auto_set_font_size(False)
        table_dates.set_fontsize(11)
        table_dates.scale(1.2, 1.8)
        
        for j in range(len(col_labels)):
            table_dates[(0, j)].set_facecolor('#2C3E50')
            table_dates[(0, j)].set_text_props(color='white', fontweight='bold')
        
        for i in range(1, len(table_data_dates) + 1):
            for j in range(len(col_labels)):
                if i % 2 == 0:
                    table_dates[(i, j)].set_facecolor('#ECF0F1')
                else:
                    table_dates[(i, j)].set_facecolor('#FFFFFF')
        
        ax.set_title('Table 4: Aligned Breakpoint Dates (Top Hives, 24-hour window)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        SAVE_PATH = os.path.join(GRAPH_FOLDER, "aligned_breakpoint_dates.png")
        plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   Table saved: {SAVE_PATH}")
    else:
        print("   No aligned dates found for top hives.")
else:
    print("   No hives with alignments found.")

# -----------------------------------------------------
# SAVE CSV SUMMARY
# -----------------------------------------------------

print("\nSaving CSV Summary...")

summary_file = os.path.join(OUTPUT_FOLDER, "pelt_summary_with_swarming.csv")
breakpoint_summary.to_csv(summary_file, index=False)
print(f"  Saved: {summary_file}")

# -----------------------------------------------------
# SUMMARY
# -----------------------------------------------------

print("\n" + "=" * 60)
print("ALL IMAGES GENERATED SUCCESSFULLY")
print("=" * 60)

print(f"\nTotal Swarming Events (Unique Dates): {total_swarming:,}")
print(f"Hives with Swarming: {hives_with_swarming}")

if total_swarming == 0:
    print("\n" + "!" * 60)
    print("WARNING: No swarming events found in the dataset!")
    print(f"The '{TARGET_COLUMN}' column has all zeros.")
    print("PELT breakpoints cannot predict swarming without swarming events.")
    print("!" * 60)
else:
    print(f"\n✓ Swarming Events Found (Unique Dates): {total_swarming}")
    print("✓ PELT + Swarming labels can now be used for prediction!")

print("\nImages saved in:")
print(f"  {GRAPH_FOLDER}")

print("\nGenerated Images:")
print("  1. pelt_breakpoint_summary.png - ALL hives with Swarming Events")
print("  2. pelt_breakpoints_barchart.png - ALL hives with Swarming indicators")
print("  3. pelt_regime_distribution.png - Regime pie + bar chart")
print("  4. pelt_overall_statistics.png - Overall statistics with Swarming")
print("  5. pelt_swarming_vs_breakpoints.png - Scatter plot with correlation")
print("  6. pelt_heatmap_clean.png - Clean heatmap (NO overlapping text)")
print("  7. pelt_per_variable_breakpoints.png - Breakpoints per variable")
print("  8. pelt_alignment_features.png - Alignment features distribution")
print("  9. breakpoint_timeline_*.png - Breakpoint timeline for hives with swarming")
print(" 10. per_variable_breakpoints_*.png - Per-variable breakpoints for hives with swarming")
print(" 11. breakpoint_alignment_24h_all_hives.png - Breakpoint alignment for ALL 48 hives (6h/12h/24h/48h windows)")
print(" 12. aligned_breakpoint_dates.png - Aligned breakpoint dates (NEW)")

print("\nCSV Files:")
print(f"  - pelt_summary_with_swarming.csv")
print(f"  - breakpoint_alignment_24h_all_hives.csv")
print(f"  - alignment_statistics_all_hives.csv")
print(f"  - aligned_breakpoint_dates.csv (NEW)")

print("\n" + "=" * 60)
print("COMPLETED")
print("=" * 60)