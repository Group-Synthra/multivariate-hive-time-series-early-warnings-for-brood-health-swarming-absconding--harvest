"""
EDA Analysis for Swarming Prediction Module
============================================
This script performs comprehensive exploratory data analysis on beehive sensor data
specifically focused on swarming prediction. It analyzes patterns, distributions,
correlations, and temporal dynamics of features related to swarming events.

Author: Swarming Prediction Team
Date: 2024
"""

from __future__ import annotations

import json
import warnings
from datetime import timedelta
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ============================================================================
# Configuration
# ============================================================================


class Config:
    """Configuration settings for EDA analysis"""

    # File paths - relative to current file location
    BASE_DIR = Path(__file__).resolve().parents[5]
    DATA_PATH = BASE_DIR / "data" / "swarming" / "hive_data_with_features.csv"
    OUTPUT_DIR = BASE_DIR / "artifacts" / "reports" / "swarming" / "eda"
    IMAGES_DIR = OUTPUT_DIR / "images"
    REPORTS_DIR = OUTPUT_DIR / "reports"

    # Analysis parameters
    TARGET_COLUMN = "swarming_label_next_72h"  # Primary target for swarming prediction
    HIVE_ID_COLUMN = "hive_id"
    TIMESTAMP_COLUMN = "timestamp"

    # Feature categories
    SENSOR_FEATURES: ClassVar[list[str]] = [
        "internal_temperature_c",
        "internal_humidity_pct",
        "co2_ppm",
        "hive_weight_kg",
        "external_temperature_c",
        "external_humidity_pct",
        "rainfall_mm_hour",
        "wind_speed_mps",
    ]

    HIVE_FEATURES: ClassVar[list[str]] = [
        "bee_stock",
        "apiary_context",
        "apiary_season",
        "nectar_flow_season_proxy",
        "dearth_season_proxy",
        "monsoon_rain_period_proxy",
    ]

    HEALTH_FEATURES: ClassVar[list[str]] = [
        "brood_health_score_0_100",
        "brood_health_label",
        "brood_health_problem_label",
    ]

    # Numeric features only (for correlation)
    NUMERIC_FEATURES = SENSOR_FEATURES + ["brood_health_score_0_100"]

    # Swarming indicators
    SWARM_INDICATORS: ClassVar[list[str]] = [
        "swarming_event_label",
        "swarming_label_next_72h",
        "absconding_event_label",
        "absconding_label_next_72h",
    ]


# Create output directories
for dir_path in [Config.OUTPUT_DIR, Config.IMAGES_DIR, Config.REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Data Loading and Initial Inspection
# ============================================================================


class DataLoader:
    """Load and perform initial inspection of the dataset"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.initial_stats = {}

    def load_data(self):
        """Load CSV data with proper parsing"""
        print(f"Loading data from: {self.file_path}")

        # Check if file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        # Load data
        self.df = pd.read_csv(self.file_path)

        # Parse timestamp
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], format="%m/%d/%Y %H:%M")

        # Sort by timestamp
        self.df = self.df.sort_values(["hive_id", "timestamp"]).reset_index(drop=True)

        print(f"✓ Loaded {len(self.df):,} records")
        print(f"✓ Time range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"✓ Number of hives: {self.df['hive_id'].nunique()}")

        return self.df

    def get_initial_stats(self):
        """Get initial dataset statistics"""
        stats = {
            "total_records": len(self.df),
            "total_hives": self.df["hive_id"].nunique(),
            "time_range": {
                "start": self.df["timestamp"].min().isoformat(),
                "end": self.df["timestamp"].max().isoformat(),
                "days": (self.df["timestamp"].max() - self.df["timestamp"].min()).days,
            },
            "bee_stocks": self.df["bee_stock"].unique().tolist(),
            "apiary_contexts": self.df["apiary_context"].unique().tolist(),
            "missing_values": self.df.isnull().sum().to_dict(),
        }
        return stats


# ============================================================================
# Data Quality Assessment
# ============================================================================


class DataQualityAnalyzer:
    """Analyze data quality, missing values, and outliers"""

    def __init__(self, df):
        self.df = df
        self.quality_report = {}

    def analyze_missing_values(self):
        """Analyze missing value patterns"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100

        missing_df = pd.DataFrame(
            {
                "Column": missing.index,
                "Missing_Count": missing.values,
                "Missing_Percentage": missing_pct.values,
            }
        )
        missing_df = missing_df[missing_df["Missing_Count"] > 0].sort_values(
            "Missing_Count", ascending=False
        )

        self.quality_report["missing_values"] = missing_df.to_dict("records")
        return missing_df

    def analyze_outliers(self, features, method="iqr"):
        """Detect outliers using IQR or Z-score method"""
        outlier_results = {}

        for feature in features:
            if feature in self.df.columns:
                # Skip if not numeric
                if not pd.api.types.is_numeric_dtype(self.df[feature]):
                    continue

                data = self.df[feature].dropna()

                if len(data) > 0 and method == "iqr":
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = data[(data < lower_bound) | (data > upper_bound)]

                    outlier_results[feature] = {
                        "count": len(outliers),
                        "percentage": (len(outliers) / len(data)) * 100,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "min": data.min(),
                        "max": data.max(),
                    }

        self.quality_report["outliers"] = outlier_results
        return outlier_results

    def get_data_types_summary(self):
        """Summarize data types"""
        types_summary = {
            "numeric": self.df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical": self.df.select_dtypes(include=["object"]).columns.tolist(),
            "datetime": self.df.select_dtypes(include=["datetime64"]).columns.tolist(),
        }
        self.quality_report["data_types"] = types_summary
        return types_summary


# ============================================================================
# Feature Distribution Analysis
# ============================================================================


class DistributionAnalyzer:
    """Analyze distributions of features"""

    def __init__(self, df, output_dir):
        self.df = df
        self.output_dir = output_dir
        self.distribution_stats = {}

    def analyze_numeric_features(self, features):
        """Analyze distribution of numeric features"""
        stats = {}

        for feature in features:
            if feature in self.df.columns:
                # Skip if not numeric
                if not pd.api.types.is_numeric_dtype(self.df[feature]):
                    continue

                data = self.df[feature].dropna()

                if len(data) > 0:
                    stats[feature] = {
                        "mean": float(data.mean()),
                        "median": float(data.median()),
                        "std": float(data.std()),
                        "skewness": float(data.skew()),
                        "kurtosis": float(data.kurtosis()),
                        "min": float(data.min()),
                        "max": float(data.max()),
                        "q25": float(data.quantile(0.25)),
                        "q75": float(data.quantile(0.75)),
                    }

        self.distribution_stats["numeric"] = stats
        return stats

    def analyze_categorical_features(self, features):
        """Analyze distribution of categorical features"""
        stats = {}

        for feature in features:
            if feature in self.df.columns:
                value_counts = self.df[feature].value_counts()
                stats[feature] = {
                    "unique_values": len(value_counts),
                    "value_counts": value_counts.to_dict(),
                    "mode": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "mode_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                }

        self.distribution_stats["categorical"] = stats
        return stats

    def plot_distributions(self, features, save=True):
        """Plot distribution of numeric features"""
        # Filter to only numeric features
        numeric_features = [
            f
            for f in features
            if f in self.df.columns and pd.api.types.is_numeric_dtype(self.df[f])
        ]

        if len(numeric_features) == 0:
            print("No numeric features found for distribution plots")
            return None

        n_features = len(numeric_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_features == 1 else axes.flatten()
        else:
            axes = axes.flatten()

        for idx, feature in enumerate(numeric_features):
            if idx < len(axes):
                ax = axes[idx]
                data = self.df[feature].dropna()

                if len(data) > 0:
                    # Histogram with KDE
                    sns.histplot(data, kde=True, ax=ax)
                    ax.set_title(f"Distribution of {feature}")
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Frequency")

                    # Add statistics annotation
                    stats_text = (
                        f"Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nSkew: {data.skew():.2f}"
                    )
                    ax.text(
                        0.95,
                        0.95,
                        stats_text,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                    )

        # Remove empty subplots
        for idx in range(len(numeric_features), len(axes)):
            if idx < len(axes):
                fig.delaxes(axes[idx])

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "feature_distribution.png", dpi=300, bbox_inches="tight")

        plt.close(fig)
        return fig


# ============================================================================
# Swarming Event Analysis
# ============================================================================


class SwarmingAnalyzer:
    """Analyze swarming events and their patterns"""

    def __init__(self, df, output_dir):
        self.df = df
        self.output_dir = output_dir
        self.swarm_analysis = {}

    def analyze_swarm_distribution(self):
        """Analyze distribution of swarming events"""
        # Swarming event counts
        swarm_counts = self.df["swarming_event_label"].value_counts()
        swarm_72h_counts = self.df["swarming_label_next_72h"].value_counts()

        # Calculate event rates
        total_records = len(self.df)

        analysis = {
            "swarming_event": {
                "count": {
                    "positive": int(swarm_counts.get(1, 0)),
                    "negative": int(swarm_counts.get(0, 0)),
                },
                "rate": {
                    "positive": float(swarm_counts.get(1, 0) / total_records * 100),
                    "negative": float(swarm_counts.get(0, 0) / total_records * 100),
                },
            },
            "swarming_72h": {
                "count": {
                    "positive": int(swarm_72h_counts.get(1, 0)),
                    "negative": int(swarm_72h_counts.get(0, 0)),
                },
                "rate": {
                    "positive": float(swarm_72h_counts.get(1, 0) / total_records * 100),
                    "negative": float(swarm_72h_counts.get(0, 0) / total_records * 100),
                },
            },
        }

        self.swarm_analysis["distribution"] = analysis
        return analysis

    def analyze_swarm_by_hive(self):
        """Analyze swarming events by hive"""
        hive_swarm = (
            self.df.groupby("hive_id")
            .agg(
                {
                    "swarming_event_label": "sum",
                    "swarming_label_next_72h": "sum",
                    "record_id": "count",
                }
            )
            .reset_index()
        )

        hive_swarm.columns = ["hive_id", "swarm_events", "swarm_next_72h", "total_records"]
        hive_swarm["swarm_rate"] = (hive_swarm["swarm_events"] / hive_swarm["total_records"]) * 100

        # Sort by swarm rate
        hive_swarm = hive_swarm.sort_values("swarm_rate", ascending=False)

        self.swarm_analysis["by_hive"] = hive_swarm.to_dict("records")
        return hive_swarm

    def analyze_swarm_by_bee_stock(self):
        """Analyze swarming events by bee stock type"""
        stock_swarm = (
            self.df.groupby("bee_stock")
            .agg(
                {
                    "swarming_event_label": "sum",
                    "swarming_label_next_72h": "sum",
                    "record_id": "count",
                }
            )
            .reset_index()
        )

        stock_swarm.columns = ["bee_stock", "swarm_events", "swarm_next_72h", "total_records"]
        stock_swarm["swarm_rate"] = (
            stock_swarm["swarm_events"] / stock_swarm["total_records"]
        ) * 100

        self.swarm_analysis["by_bee_stock"] = stock_swarm.to_dict("records")
        return stock_swarm

    def plot_swarm_analysis(self, save=True):
        """Plot swarming analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Swarm distribution
        ax1 = axes[0, 0]
        swarm_data = self.df["swarming_label_next_72h"].value_counts()
        labels = ["No Swarm (0)", "Swarm (1)"]
        ax1.pie(
            swarm_data.values, labels=labels, autopct="%1.1f%%", explode=[0, 0.05], startangle=90
        )
        ax1.set_title("Swarming Event Distribution (Next 72h)")

        # 2. Swarm by hive
        ax2 = axes[0, 1]
        hive_swarm = self.df.groupby("hive_id")["swarming_label_next_72h"].mean() * 100
        hive_swarm.sort_values(ascending=False).head(10).plot(kind="bar", ax=ax2)
        ax2.set_title("Top 10 Hives by Swarming Rate (Next 72h)")
        ax2.set_xlabel("Hive ID")
        ax2.set_ylabel("Swarming Rate (%)")
        ax2.tick_params(axis="x", rotation=45)

        # 3. Swarm by bee stock
        ax3 = axes[1, 0]
        stock_swarm = self.df.groupby("bee_stock")["swarming_label_next_72h"].mean() * 100
        stock_swarm.plot(kind="bar", ax=ax3, color=["#2ecc71", "#3498db", "#e74c3c"])
        ax3.set_title("Swarming Rate by Bee Stock Type")
        ax3.set_xlabel("Bee Stock")
        ax3.set_ylabel("Swarming Rate (%)")
        ax3.tick_params(axis="x", rotation=45)

        # 4. Temporal swarming pattern
        ax4 = axes[1, 1]
        daily_swarm = (
            self.df.set_index("timestamp")
            .groupby(self.df["timestamp"].dt.date)["swarming_label_next_72h"]
            .mean()
            * 100
        )
        daily_swarm.plot(ax=ax4, color="#e67e22")
        ax4.set_title("Daily Swarming Rate Over Time")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Swarming Rate (%)")
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "swarm_indicators.png", dpi=300, bbox_inches="tight")

        plt.close(fig)
        return fig


# ============================================================================
# Correlation Analysis
# ============================================================================


class CorrelationAnalyzer:
    """Analyze correlations between features and swarming"""

    def __init__(self, df, output_dir):
        self.df = df
        self.output_dir = output_dir
        self.correlation_results = {}

    def compute_correlations(self, features, target):
        """Compute correlations with target variable"""
        correlations = {}

        for feature in features:
            if feature in self.df.columns and feature != target:
                # Skip if not numeric
                if not pd.api.types.is_numeric_dtype(self.df[feature]):
                    continue

                # Pearson correlation
                pearson_corr = self.df[feature].corr(self.df[target])

                # Spearman correlation (for non-linear relationships)
                spearman_corr = self.df[feature].corr(self.df[target], method="spearman")

                if not np.isnan(pearson_corr):
                    correlations[feature] = {
                        "pearson": float(pearson_corr),
                        "spearman": float(spearman_corr),
                        "abs_pearson": float(abs(pearson_corr)),
                    }

        # Sort by absolute Pearson correlation
        self.correlation_results["feature_correlations"] = sorted(
            correlations.items(), key=lambda x: x[1]["abs_pearson"], reverse=True
        )

        return self.correlation_results

    def compute_feature_correlation_matrix(self, features):
        """Compute correlation matrix between features"""
        # Select numeric features only
        numeric_features = []
        for f in features:
            if f in self.df.columns and pd.api.types.is_numeric_dtype(self.df[f]):
                numeric_features.append(f)

        if len(numeric_features) > 1:
            correlation_matrix = self.df[numeric_features].corr()
            self.correlation_results["feature_matrix"] = correlation_matrix.to_dict()
            return correlation_matrix

        return None

    def plot_correlation_matrix(self, features, save=True):
        """Plot correlation matrix heatmap"""
        # Select numeric features only
        numeric_features = []
        for f in features:
            if f in self.df.columns and pd.api.types.is_numeric_dtype(self.df[f]):
                numeric_features.append(f)

        if len(numeric_features) < 2:
            print("Not enough numeric features for correlation matrix")
            return None

        # Compute correlation matrix
        corr_matrix = self.df[numeric_features].corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 12))

        # Mask upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            center=0,
            ax=ax,
            square=True,
            linewidths=0.5,
        )

        ax.set_title("Feature Correlation Matrix", fontsize=16)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight")

        plt.close(fig)
        return fig

    def plot_top_correlations(self, target, features, top_n=10, save=True):
        """Plot top correlated features with target"""
        # Compute correlations
        correlations = []
        for feature in features:
            if feature in self.df.columns and feature != target:
                # Skip if not numeric
                if not pd.api.types.is_numeric_dtype(self.df[feature]):
                    continue
                corr = self.df[feature].corr(self.df[target])
                if not np.isnan(corr):
                    correlations.append((feature, corr))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = correlations[:top_n]

        if len(top_features) == 0:
            print("No valid correlations found")
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        features_names = [f[0] for f in top_features]
        corr_values = [f[1] for f in top_features]
        colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in corr_values]

        ax.barh(features_names, corr_values, color=colors)
        ax.set_xlabel("Correlation Coefficient")
        ax.set_title(f"Top {top_n} Features Correlated with {target}")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "top_correlations.png", dpi=300, bbox_inches="tight")

        plt.close(fig)
        return fig


# ============================================================================
# Temporal Pattern Analysis
# ============================================================================


class TemporalAnalyzer:
    """Analyze temporal patterns in data"""

    def __init__(self, df, output_dir):
        self.df = df
        self.output_dir = output_dir
        self.temporal_results = {}

    def analyze_by_hour(self, feature):
        """Analyze feature by hour of day"""
        self.df["hour"] = self.df["timestamp"].dt.hour

        hourly_mean = self.df.groupby("hour")[feature].mean()
        hourly_std = self.df.groupby("hour")[feature].std()

        self.temporal_results["hourly"] = {
            "mean": hourly_mean.to_dict(),
            "std": hourly_std.to_dict(),
        }

        return hourly_mean, hourly_std

    def analyze_by_day(self, feature):
        """Analyze feature by day of week"""
        self.df["day_of_week"] = self.df["timestamp"].dt.dayofweek
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        daily_mean = self.df.groupby("day_of_week")[feature].mean()
        daily_std = self.df.groupby("day_of_week")[feature].std()

        # Convert index to day names
        daily_mean.index = [day_names[i] for i in daily_mean.index]
        daily_std.index = [day_names[i] for i in daily_std.index]

        self.temporal_results["daily"] = {"mean": daily_mean.to_dict(), "std": daily_std.to_dict()}

        return daily_mean, daily_std

    def plot_temporal_patterns(self, features, save=True):
        """Plot temporal patterns for features"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Hourly patterns - filter to numeric features
        ax1 = axes[0, 0]
        colors = ["#2ecc71", "#3498db", "#e74c3c"]
        plot_features = []
        for f in features[:3]:
            if f in self.df.columns and pd.api.types.is_numeric_dtype(self.df[f]):
                plot_features.append(f)

        for idx, feature in enumerate(plot_features):
            hourly_mean = self.df.groupby(self.df["timestamp"].dt.hour)[feature].mean()
            ax1.plot(
                hourly_mean.index,
                hourly_mean.values,
                label=feature,
                marker="o",
                color=colors[idx % len(colors)],
            )
        ax1.set_title("Hourly Patterns of Key Features")
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Average Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Feature comparison: Swarm vs No Swarm
        ax2 = axes[0, 1]
        if len(plot_features) > 0:
            swarm_status = self.df.groupby("swarming_label_next_72h")[plot_features].mean()

            if len(swarm_status) > 0:
                x = np.arange(len(plot_features))
                width = 0.35

                for i, status in enumerate(swarm_status.index):
                    values = swarm_status.loc[status].values
                    label = "Swarm" if status == 1 else "No Swarm"
                    offset = -width / 2 if status == 1 else width / 2
                    ax2.bar(x + offset, values, width, label=label)

                ax2.set_title("Feature Comparison: Swarm vs No Swarm")
                ax2.set_xlabel("Feature")
                ax2.set_ylabel("Mean Value")
                ax2.set_xticks(x)
                ax2.set_xticklabels(
                    [f.replace("_", " ").title() for f in plot_features], rotation=45
                )
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        # 3. Sample time series
        ax3 = axes[1, 0]
        sample_hive = self.df["hive_id"].unique()[0]
        hive_data = self.df[self.df["hive_id"] == sample_hive]

        # Select sample dates
        start_date = hive_data["timestamp"].min()
        end_date = start_date + timedelta(days=7)
        sample_data = hive_data[
            (hive_data["timestamp"] >= start_date) & (hive_data["timestamp"] <= end_date)
        ]

        if len(sample_data) > 0:
            ax3_twin = ax3.twinx()

            # Plot temperature and CO2
            ax3.plot(
                sample_data["timestamp"],
                sample_data["internal_temperature_c"],
                color="blue",
                label="Temperature (°C)",
            )
            ax3_twin.plot(
                sample_data["timestamp"], sample_data["co2_ppm"], color="red", label="CO2 (ppm)"
            )

            # Mark swarming events
            swarm_indices = sample_data[sample_data["swarming_label_next_72h"] == 1].index
            for idx in swarm_indices:
                ax3.axvline(
                    x=sample_data.loc[idx, "timestamp"],
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1,
                )

            ax3.set_title(f"Hive {sample_hive} - 7 Day Sample (Red lines: swarming events)")
            ax3.set_xlabel("Timestamp")
            ax3.set_ylabel("Temperature (°C)")
            ax3_twin.set_ylabel("CO2 (ppm)")
            ax3.legend(loc="upper left")
            ax3_twin.legend(loc="upper right")
            ax3.grid(True, alpha=0.3)

        # 4. Seasonal patterns
        ax4 = axes[1, 1]

        # Create season groups
        season_groups = self.df.groupby("apiary_season").agg(
            {
                "internal_temperature_c": "mean",
                "internal_humidity_pct": "mean",
                "co2_ppm": "mean",
                "swarming_label_next_72h": "mean",
            }
        )

        if len(season_groups) > 0:
            season_groups[["internal_temperature_c", "internal_humidity_pct", "co2_ppm"]].plot(
                kind="bar", ax=ax4
            )
            ax4.set_title("Seasonal Patterns of Key Features")
            ax4.set_xlabel("Season")
            ax4.set_ylabel("Average Value")
            ax4.legend()
            ax4.tick_params(axis="x", rotation=45)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "temporal_patterns.png", dpi=300, bbox_inches="tight")

        plt.close(fig)
        return fig


# ============================================================================
# PELT Change Point Detection (Simplified)
# ============================================================================


class PELTAnalyzer:
    """Perform PELT change point detection for regime analysis (simplified version)"""

    def __init__(self, df, output_dir):
        self.df = df
        self.output_dir = output_dir
        self.pelt_results = {}

    def simple_change_point_detection(self, data, threshold=2.5):
        """Simple change point detection using z-score"""
        if len(data) < 10:
            return []

        mean = data.mean()
        std = data.std()

        if std == 0:
            return []

        z_scores = np.abs((data - mean) / std)
        change_points = np.where(z_scores > threshold)[0]

        return change_points.tolist()

    def detect_change_points(self, hive_id, feature):
        """Detect change points using simple z-score method"""
        # Get data for specific hive
        hive_data = self.df[self.df["hive_id"] == hive_id].sort_values("timestamp")

        if len(hive_data) < 50:
            print(f"Warning: Hive {hive_id} has insufficient data for change point analysis")
            return None

        # Skip if not numeric
        if not pd.api.types.is_numeric_dtype(hive_data[feature]):
            return None

        # Extract feature values
        values = hive_data[feature].values

        # Detect change points
        change_points = self.simple_change_point_detection(values)

        # Convert to timestamps
        change_timestamps = [hive_data.iloc[i]["timestamp"] for i in change_points]

        return {
            "hive_id": hive_id,
            "feature": feature,
            "change_points": change_points,
            "change_timestamps": change_timestamps,
            "n_segments": len(change_points) + 1,
        }

    def analyze_regime_segments(self, hive_id, feature, change_point_result):
        """Analyze segments between change points"""
        if change_point_result is None:
            return None

        hive_data = self.df[self.df["hive_id"] == hive_id].sort_values("timestamp")
        change_points = change_point_result["change_points"]

        # Add start and end points
        all_points = [0] + change_points + [len(hive_data)]

        segments = []
        for i in range(len(all_points) - 1):
            start_idx = all_points[i]
            end_idx = all_points[i + 1]

            segment_data = hive_data.iloc[start_idx:end_idx]

            if len(segment_data) > 0:
                segments.append(
                    {
                        "segment_id": i,
                        "start_time": segment_data["timestamp"].iloc[0].isoformat(),
                        "end_time": segment_data["timestamp"].iloc[-1].isoformat(),
                        "duration_hours": (
                            segment_data["timestamp"].iloc[-1] - segment_data["timestamp"].iloc[0]
                        ).total_seconds()
                        / 3600,
                        "mean_value": float(segment_data[feature].mean()),
                        "std_value": float(segment_data[feature].std()),
                        "swarm_ratio": float(segment_data["swarming_label_next_72h"].mean()),
                    }
                )

        return {
            "hive_id": hive_id,
            "feature": feature,
            "segments": segments,
            "total_segments": len(segments),
        }

    def plot_pelt_results(self, hive_id, feature, save=True):
        """Plot change point detection results"""
        result = self.detect_change_points(hive_id, feature)

        if result is None:
            return None

        # Get data
        hive_data = self.df[self.df["hive_id"] == hive_id].sort_values("timestamp")
        change_points = result["change_points"]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: Feature with change points
        ax1.plot(
            hive_data["timestamp"], hive_data[feature], label=feature, alpha=0.7, color="#3498db"
        )

        # Mark change points
        for cp_idx in change_points:
            ax1.axvline(
                x=hive_data.iloc[cp_idx]["timestamp"],
                color="red",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
            )

        # Mark swarming events
        swarm_indices = hive_data[hive_data["swarming_label_next_72h"] == 1].index
        for idx in swarm_indices:
            ax1.axvline(
                x=hive_data.loc[idx, "timestamp"],
                color="orange",
                linestyle="-",
                alpha=0.3,
                linewidth=2,
            )

        ax1.set_title(f"Hive {hive_id} - {feature} with Change Points")
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel(feature)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Regime segments
        segments = self.analyze_regime_segments(hive_id, feature, result)

        if segments and len(segments["segments"]) > 0:
            segment_data = pd.DataFrame(segments["segments"])

            # Plot segment means
            ax2.bar(
                range(len(segment_data)), segment_data["mean_value"], alpha=0.7, color="#2ecc71"
            )

            # Add error bars for std
            ax2.errorbar(
                range(len(segment_data)),
                segment_data["mean_value"],
                yerr=segment_data["std_value"],
                fmt="none",
                ecolor="red",
                capsize=5,
            )

            ax2.set_title("Regime Segments - Mean and Standard Deviation")
            ax2.set_xlabel("Segment Number")
            ax2.set_ylabel(f"{feature} Value")
            ax2.grid(True, alpha=0.3)

            # Add swarm ratio as text
            for i, row in segment_data.iterrows():
                ax2.text(
                    i,
                    row["mean_value"] + row["std_value"] * 0.5,
                    f"Swarm: {row['swarm_ratio'] * 100:.1f}%",
                    ha="center",
                    fontsize=8,
                )

        plt.tight_layout()

        if save:
            plt.savefig(
                self.output_dir / f"pelt_regime_hive_{hive_id}.png", dpi=300, bbox_inches="tight"
            )

        plt.close(fig)
        return fig

    def summarize_pelt_analysis(self):
        """Summarize change point analysis across hives"""
        summary = {}

        for hive_id in self.df["hive_id"].unique()[:3]:  # Limit to first 3 hives
            hive_summary = {}

            for feature in ["internal_temperature_c", "co2_ppm", "hive_weight_kg"]:
                if feature in self.df.columns and pd.api.types.is_numeric_dtype(self.df[feature]):
                    result = self.detect_change_points(hive_id, feature)
                    if result:
                        segments = self.analyze_regime_segments(hive_id, feature, result)
                        if segments and len(segments["segments"]) > 0:
                            avg_duration = np.mean(
                                [s["duration_hours"] for s in segments["segments"]]
                            )
                            hive_summary[feature] = {
                                "n_segments": segments["total_segments"],
                                "avg_segment_duration": float(avg_duration),
                            }

            summary[f"hive_{hive_id}"] = hive_summary

        self.pelt_results["summary"] = summary
        return summary


# ============================================================================
# Feature Engineering Insights
# ============================================================================


class FeatureEngineeringAnalyzer:
    """Analyze potential feature engineering opportunities"""

    def __init__(self, df):
        self.df = df
        self.feature_insights = {}

    def analyze_rate_of_change(self, feature):
        """Analyze rate of change patterns"""
        # Skip if not numeric
        if not pd.api.types.is_numeric_dtype(self.df[feature]):
            return None

        # Calculate rate of change
        self.df[f"{feature}_roc"] = self.df.groupby("hive_id")[feature].diff()

        # Analyze by swarming status
        roc_stats = self.df.groupby("swarming_label_next_72h")[f"{feature}_roc"].agg(
            ["mean", "std", "median"]
        )

        self.feature_insights[f"{feature}_roc"] = roc_stats.to_dict()
        return roc_stats

    def analyze_rolling_stats(self, feature, window=24):
        """Analyze rolling statistics"""
        # Skip if not numeric
        if not pd.api.types.is_numeric_dtype(self.df[feature]):
            return None, None

        # Calculate rolling statistics
        self.df[f"{feature}_rolling_mean"] = self.df.groupby("hive_id")[feature].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        self.df[f"{feature}_rolling_std"] = self.df.groupby("hive_id")[feature].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )

        # Analyze by swarming status
        rolling_mean_stats = self.df.groupby("swarming_label_next_72h")[
            f"{feature}_rolling_mean"
        ].agg(["mean", "std"])
        rolling_std_stats = self.df.groupby("swarming_label_next_72h")[
            f"{feature}_rolling_std"
        ].agg(["mean", "std"])

        self.feature_insights[f"{feature}_rolling"] = {
            "mean": rolling_mean_stats.to_dict(),
            "std": rolling_std_stats.to_dict(),
        }

        return rolling_mean_stats, rolling_std_stats

    def suggest_features(self):
        """Suggest new features based on analysis"""
        suggestions = []

        # Temporal features
        suggestions.append(
            {
                "name": "hour_of_day",
                "description": "Hour of day (0-23) - captures diurnal patterns",
                "type": "temporal",
            }
        )
        suggestions.append(
            {
                "name": "day_of_week",
                "description": "Day of week (0-6) - captures weekly patterns",
                "type": "temporal",
            }
        )

        # Rate of change features
        for feature in ["internal_temperature_c", "co2_ppm", "hive_weight_kg"]:
            suggestions.append(
                {
                    "name": f"{feature}_roc",
                    "description": f"Rate of change of {feature}",
                    "type": "derived",
                }
            )

        # Rolling statistics
        for feature in ["internal_temperature_c", "co2_ppm", "hive_weight_kg"]:
            suggestions.append(
                {
                    "name": f"{feature}_rolling_mean_24h",
                    "description": f"24-hour rolling mean of {feature}",
                    "type": "rolling",
                }
            )
            suggestions.append(
                {
                    "name": f"{feature}_rolling_std_24h",
                    "description": f"24-hour rolling standard deviation of {feature}",
                    "type": "rolling",
                }
            )

        # Deviation features
        for feature in ["internal_temperature_c", "co2_ppm", "hive_weight_kg"]:
            suggestions.append(
                {
                    "name": f"{feature}_deviation",
                    "description": f"Deviation from normal baseline for {feature}",
                    "type": "derived",
                }
            )

        # Regime-based features
        suggestions.append(
            {
                "name": "regime_stability_score",
                "description": "Stability score based on change point analysis",
                "type": "regime",
            }
        )
        suggestions.append(
            {
                "name": "time_since_last_change_point",
                "description": "Time in hours since last change point",
                "type": "regime",
            }
        )

        self.feature_insights["suggestions"] = suggestions
        return suggestions


# ============================================================================
# Main EDA Runner
# ============================================================================


class EDARunner:
    """Main EDA runner that orchestrates all analysis"""

    def __init__(self):
        self.loader = DataLoader(Config.DATA_PATH)
        self.quality_analyzer = None
        self.distribution_analyzer = None
        self.swarming_analyzer = None
        self.correlation_analyzer = None
        self.temporal_analyzer = None
        self.pelt_analyzer = None
        self.feature_analyzer = None

        self.results = {}

    def run_all_analyses(self):
        """Run all EDA analyses"""
        print("=" * 80)
        print("SWARMING PREDICTION MODULE - EDA ANALYSIS")
        print("=" * 80)

        # 1. Load data
        print("\n[1/8] Loading data...")
        df = self.loader.load_data()
        self.results["initial_stats"] = self.loader.get_initial_stats()

        # 2. Data quality analysis
        print("\n[2/8] Analyzing data quality...")
        self.quality_analyzer = DataQualityAnalyzer(df)
        self.quality_analyzer.analyze_missing_values()
        self.quality_analyzer.analyze_outliers(Config.SENSOR_FEATURES)
        self.quality_analyzer.get_data_types_summary()
        self.results["data_quality"] = self.quality_analyzer.quality_report

        # 3. Distribution analysis
        print("\n[3/8] Analyzing feature distributions...")
        self.distribution_analyzer = DistributionAnalyzer(df, Config.IMAGES_DIR)
        self.distribution_analyzer.analyze_numeric_features(Config.SENSOR_FEATURES)
        self.distribution_analyzer.analyze_categorical_features(
            ["bee_stock", "apiary_context", "apiary_season", "brood_health_label"]
        )

        # Plot distributions
        self.distribution_analyzer.plot_distributions(Config.SENSOR_FEATURES[:6], save=True)
        self.results["distribution_stats"] = self.distribution_analyzer.distribution_stats

        # 4. Swarming analysis
        print("\n[4/8] Analyzing swarming events...")
        self.swarming_analyzer = SwarmingAnalyzer(df, Config.IMAGES_DIR)
        self.swarming_analyzer.analyze_swarm_distribution()
        self.swarming_analyzer.analyze_swarm_by_hive()
        self.swarming_analyzer.analyze_swarm_by_bee_stock()

        # Plot swarm analysis
        self.swarming_analyzer.plot_swarm_analysis(save=True)
        self.results["swarming_analysis"] = self.swarming_analyzer.swarm_analysis

        # 5. Correlation analysis
        print("\n[5/8] Analyzing correlations...")
        self.correlation_analyzer = CorrelationAnalyzer(df, Config.IMAGES_DIR)
        self.correlation_analyzer.compute_correlations(
            Config.NUMERIC_FEATURES,  # Use only numeric features
            Config.TARGET_COLUMN,
        )
        self.correlation_analyzer.compute_feature_correlation_matrix(Config.NUMERIC_FEATURES)

        # Plot correlations
        self.correlation_analyzer.plot_correlation_matrix(Config.NUMERIC_FEATURES, save=True)
        self.correlation_analyzer.plot_top_correlations(
            Config.TARGET_COLUMN, Config.NUMERIC_FEATURES, top_n=10, save=True
        )
        self.results["correlation_analysis"] = self.correlation_analyzer.correlation_results

        # 6. Temporal analysis
        print("\n[6/8] Analyzing temporal patterns...")
        self.temporal_analyzer = TemporalAnalyzer(df, Config.IMAGES_DIR)

        # Analyze key features
        for feature in ["internal_temperature_c", "co2_ppm", "hive_weight_kg"]:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                _hourly_mean, _hourly_std = self.temporal_analyzer.analyze_by_hour(feature)
                _daily_mean, _daily_std = self.temporal_analyzer.analyze_by_day(feature)

        # Plot temporal patterns
        self.temporal_analyzer.plot_temporal_patterns(
            ["internal_temperature_c", "co2_ppm", "hive_weight_kg"], save=True
        )
        self.results["temporal_analysis"] = self.temporal_analyzer.temporal_results

        # 7. PELT analysis (simplified)
        print("\n[7/8] Performing change point detection...")
        self.pelt_analyzer = PELTAnalyzer(df, Config.IMAGES_DIR)

        # Analyze sample hives
        hive_ids = df["hive_id"].unique()[:3]
        for hive_id in hive_ids:
            for feature in ["internal_temperature_c", "co2_ppm", "hive_weight_kg"]:
                if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                    self.pelt_analyzer.plot_pelt_results(hive_id, feature, save=True)

        self.pelt_analyzer.summarize_pelt_analysis()
        self.results["pelt_analysis"] = self.pelt_analyzer.pelt_results

        # 8. Feature engineering suggestions
        print("\n[8/8] Generating feature engineering suggestions...")
        self.feature_analyzer = FeatureEngineeringAnalyzer(df)

        # Analyze rate of change
        for feature in ["internal_temperature_c", "co2_ppm", "hive_weight_kg"]:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                self.feature_analyzer.analyze_rate_of_change(feature)

        # Analyze rolling statistics
        for feature in ["internal_temperature_c", "co2_ppm", "hive_weight_kg"]:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                _mean_stats, _std_stats = self.feature_analyzer.analyze_rolling_stats(
                    feature, window=24
                )

        # Get feature suggestions
        suggestions = self.feature_analyzer.suggest_features()
        self.results["feature_suggestions"] = suggestions

        # 9. Generate summary report
        print("\n" + "=" * 80)
        print("EDA COMPLETE - SUMMARY")
        print("=" * 80)
        self.print_summary()

        # 10. Save results
        self.save_results()

        return self.results

    def print_summary(self):
        """Print summary of EDA findings"""
        print("\n📊 Dataset Overview:")
        print(f"   - Total Records: {self.results['initial_stats']['total_records']:,}")
        print(f"   - Total Hives: {self.results['initial_stats']['total_hives']}")
        print(f"   - Time Range: {self.results['initial_stats']['time_range']['days']} days")
        print(f"   - Bee Stocks: {', '.join(self.results['initial_stats']['bee_stocks'])}")

        print("\n📈 Swarming Events:")
        swarm_dist = self.results.get("swarming_analysis", {}).get("distribution", {})
        if swarm_dist:
            positive_rate = swarm_dist.get("swarming_72h", {}).get("rate", {}).get("positive", 0)
            print(f"   - Swarm Rate (Next 72h): {positive_rate:.2f}%")

        print("\n🔍 Key Correlations with Swarming:")
        correlations = self.results.get("correlation_analysis", {}).get("feature_correlations", [])
        for feature, corr in correlations[:5]:
            corr_value = corr.get("pearson", 0)
            print(f"   - {feature}: {corr_value:.3f}")

        print("\n📐 Change Point Analysis Summary:")
        pelt_summary = self.results.get("pelt_analysis", {}).get("summary", {})
        for hive, summary in pelt_summary.items():
            print(f"   - {hive}: {len(summary)} features analyzed")

        print(f"\n💡 Suggested New Features: {len(self.results.get('feature_suggestions', []))}")

        print(f"\n📁 Output saved to: {Config.OUTPUT_DIR}")

    def save_results(self):
        """Save results to JSON and CSV files"""
        # Save dashboard JSON
        dashboard_data = {
            "summary": {
                "total_records": self.results["initial_stats"]["total_records"],
                "total_hives": self.results["initial_stats"]["total_hives"],
                "time_days": self.results["initial_stats"]["time_range"]["days"],
                "swarm_rate": self.results.get("swarming_analysis", {})
                .get("distribution", {})
                .get("swarming_72h", {})
                .get("rate", {})
                .get("positive", 0),
            },
            "data_quality": self.results.get("data_quality", {}),
            "distribution_stats": self.results.get("distribution_stats", {}),
            "swarming_analysis": self.results.get("swarming_analysis", {}),
            "correlation_analysis": {
                "top_features": [
                    {"feature": f, "correlation": c.get("pearson", 0)}
                    for f, c in self.results.get("correlation_analysis", {}).get(
                        "feature_correlations", []
                    )[:10]
                ]
            },
            "feature_suggestions": self.results.get("feature_suggestions", []),
        }

        # Save JSON
        with open(Config.OUTPUT_DIR / "dashboard.json", "w") as f:
            json.dump(dashboard_data, f, indent=2, default=str)

        # Save data quality report
        missing_df = pd.DataFrame(self.results.get("data_quality", {}).get("missing_values", []))
        if not missing_df.empty:
            missing_df.to_csv(Config.REPORTS_DIR / "data_quality_report.csv", index=False)

        # Save feature analysis summary
        with open(Config.REPORTS_DIR / "feature_analysis_summary.txt", "w") as f:
            f.write("Swarming Prediction Module - Feature Analysis Summary\n")
            f.write("=" * 60 + "\n\n")

            f.write("Top Correlated Features:\n")
            correlations = self.results.get("correlation_analysis", {}).get(
                "feature_correlations", []
            )
            f.writelines(
                f"  - {feature}: {corr.get('pearson', 0):.4f}\n"
                for feature, corr in correlations[:10]
            )

            f.write("\nSuggested New Features:\n")
            for suggestion in self.results.get("feature_suggestions", []):
                f.write(f"  - {suggestion['name']}: {suggestion['description']}\n")
                f.write(f"    Type: {suggestion['type']}\n")

        print(f"\n✓ Results saved to {Config.OUTPUT_DIR}")


# ============================================================================
# Execution
# ============================================================================


def main():
    """Run the swarming-only exploratory data analysis pipeline."""
    # Create output directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    Config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run EDA
    runner = EDARunner()
    runner.run_all_analyses()

    print("\n" + "=" * 80)
    print("EDA ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n📁 Results saved in: {Config.OUTPUT_DIR}")
    print("   - dashboard.json: Summary data for frontend")
    print("   - images/: All visualization plots")
    print("   - reports/: Data quality and feature analysis reports")
    print("\n🚀 Next Steps:")
    print("   1. Review feature correlation results")
    print("   2. Implement suggested new features")
    print("   3. Address data quality issues")
    print("   4. Proceed to model training with enhanced feature set")


if __name__ == "__main__":
    main()
