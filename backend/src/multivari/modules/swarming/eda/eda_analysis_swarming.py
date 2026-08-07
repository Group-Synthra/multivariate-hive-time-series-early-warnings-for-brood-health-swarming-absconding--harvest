"""Research-grade exploratory analysis for the swarming module.

The module reads the swarming-prepared CSV, produces reproducible statistical
tables, a compact dashboard JSON document, and publication-style figures.  It is
deliberately independent of ``multivari.common`` so that the swarming work can be
maintained and assessed as a separate group-project contribution.

Run from ``backend`` with::

    python scripts/run_swarming_eda.py

or::

    python -m multivari.modules.swarming.eda.eda_analysis_swarming
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EDAConfig:
    """Paths and analysis variables used only by the swarming module."""

    backend_dir: Path
    data_path: Path
    output_dir: Path

    hive_column: str = "hive_id"
    timestamp_column: str = "timestamp"
    event_column: str = "swarming_event_label"
    target_column: str = "swarming_label_next_72h"

    sensor_features: ClassVar[tuple[str, ...]] = (
        "internal_temperature_c",
        "internal_humidity_pct",
        "co2_ppm",
        "hive_weight_kg",
        "external_temperature_c",
        "external_humidity_pct",
    )
    sensor_labels: ClassVar[dict[str, str]] = {
        "internal_temperature_c": "Internal temperature (°C)",
        "internal_humidity_pct": "Internal humidity (%)",
        "co2_ppm": "CO₂ concentration (ppm)",
        "hive_weight_kg": "Hive weight (kg)",
        "external_temperature_c": "External temperature (°C)",
        "external_humidity_pct": "External humidity (%)",
    }

    @classmethod
    def default(cls) -> EDAConfig:
        # .../backend/src/multivari/modules/swarming/eda/this_file.py
        resolved = Path(__file__).resolve()
        backend_dir = next(
            (parent for parent in resolved.parents if parent.name.lower() == "backend"),
            Path.cwd(),
        )
        output_dir = backend_dir / "artifacts" / "reports" / "swarming" / "eda"
        return cls(
            backend_dir=backend_dir,
            data_path=backend_dir / "data" / "swarming" / "hive_data_with_features.csv",
            output_dir=output_dir,
        )

    @property
    def images_dir(self) -> Path:
        return self.output_dir / "images"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"


def _configure_plotting() -> None:
    """Apply a restrained, journal-appropriate plotting style."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#20252b",
            "axes.titlecolor": "#17212b",
            "axes.titleweight": "semibold",
            "axes.grid": True,
            "grid.color": "#d8dde3",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "xtick.color": "#30363d",
            "ytick.color": "#30363d",
        }
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _standardized_mean_difference(negative: pd.Series, positive: pd.Series) -> float:
    """Return Cohen's d using the pooled sample standard deviation."""

    negative = pd.to_numeric(negative, errors="coerce").dropna()
    positive = pd.to_numeric(positive, errors="coerce").dropna()
    if len(negative) < 2 or len(positive) < 2:
        return float("nan")
    numerator = positive.mean() - negative.mean()
    denominator = np.sqrt(
        ((len(negative) - 1) * negative.var(ddof=1) + (len(positive) - 1) * positive.var(ddof=1))
        / (len(negative) + len(positive) - 2)
    )
    return float(numerator / denominator) if denominator > 0 else float("nan")


class SwarmingEDA:
    """Generate the complete, swarming-specific exploratory analysis."""

    NAVY = "#263b50"
    SLATE = "#657786"
    LIGHT_SLATE = "#aeb8c2"
    BURGUNDY = "#7a3343"
    CHARCOAL = "#30363d"

    def __init__(self, config: EDAConfig | None = None) -> None:
        self.config = config or EDAConfig.default()
        self.df: pd.DataFrame | None = None
        self.results: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        """Run validation, statistical analysis, exports, and visualisation."""

        _configure_plotting()
        self._prepare_directories()
        self.df = self._load_data()

        LOGGER.info("Analysing %s swarming records", f"{len(self.df):,}")
        self.results = {
            "initial_stats": self._dataset_overview(),
            "data_quality": self._data_quality(),
            "distribution_stats": self._distribution_statistics(),
            "swarming_analysis": self._swarming_analysis(),
            "correlation_analysis": self._association_analysis(),
            "temporal_analysis": self._temporal_analysis(),
            "methodology": self._methodology(),
        }
        self._export_tables()
        self._create_figures()
        self._write_dashboard()
        self._write_report()
        return self.results

    def _prepare_directories(self) -> None:
        for directory in (
            self.config.output_dir,
            self.config.images_dir,
            self.config.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        # Do not expose obsolete PELT-regime plots in the EDA gallery.
        for path in self.config.images_dir.glob("pelt_regime_hive_*.png"):
            path.unlink(missing_ok=True)

    def _load_data(self) -> pd.DataFrame:
        path = self.config.data_path
        if not path.exists():
            raise FileNotFoundError(
                f"Swarming dataset not found: {path}\n"
                "Create the module-specific dataset before running swarming EDA."
            )
        frame = pd.read_csv(path, low_memory=False)
        required = {
            self.config.hive_column,
            self.config.timestamp_column,
            self.config.event_column,
            self.config.target_column,
            *self.config.sensor_features,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError("Swarming dataset is missing required columns: " + ", ".join(missing))

        frame[self.config.timestamp_column] = pd.to_datetime(
            frame[self.config.timestamp_column], errors="coerce"
        )
        for column in (*self.config.sensor_features, self.config.event_column, self.config.target_column):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in (self.config.event_column, self.config.target_column):
            invalid = frame[column].dropna().loc[lambda values: ~values.isin([0, 1])]
            if not invalid.empty:
                raise ValueError(f"{column} must be binary (0/1); found {sorted(invalid.unique())}")
        return frame.sort_values(
            [self.config.hive_column, self.config.timestamp_column], na_position="last"
        ).reset_index(drop=True)

    @property
    def data(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("EDA data has not been loaded")
        return self.df

    def _dataset_overview(self) -> dict[str, Any]:
        timestamps = self.data[self.config.timestamp_column].dropna()
        start = timestamps.min() if not timestamps.empty else pd.NaT
        end = timestamps.max() if not timestamps.empty else pd.NaT
        return {
            "total_records": len(self.data),
            "total_hives": self.data[self.config.hive_column].nunique(dropna=True),
            "time_range": {
                "start": start,
                "end": end,
                "days": int((end - start).days) if pd.notna(start) and pd.notna(end) else 0,
            },
            "sampling_granularity": self.data.get(
                "data_granularity", pd.Series(dtype="object")
            ).dropna().astype(str).value_counts().to_dict(),
            "bee_stocks": sorted(
                self.data.get("bee_stock", pd.Series(dtype="object")).dropna().astype(str).unique()
            ),
            "apiary_contexts": sorted(
                self.data.get("apiary_context", pd.Series(dtype="object"))
                .dropna().astype(str).unique()
            ),
        }

    def _data_quality(self) -> dict[str, Any]:
        columns = [
            self.config.hive_column,
            self.config.timestamp_column,
            *self.config.sensor_features,
            self.config.event_column,
            self.config.target_column,
        ]
        missing_rows = []
        for column in columns:
            count = int(self.data[column].isna().sum())
            missing_rows.append(
                {
                    "column": column,
                    "missing_count": count,
                    "missing_percentage": count / len(self.data) * 100,
                }
            )

        outliers: dict[str, dict[str, float | int]] = {}
        for column in self.config.sensor_features:
            values = self.data[column].dropna()
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            count = int(((values < lower) | (values > upper)).sum())
            outliers[column] = {
                "count": count,
                "percentage": count / len(values) * 100 if len(values) else 0.0,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }

        duplicate_records = int(self.data.duplicated().sum())
        duplicate_hive_times = int(
            self.data.duplicated(
                [self.config.hive_column, self.config.timestamp_column], keep=False
            ).sum()
        )
        return {
            "missing_values": missing_rows,
            "outliers": outliers,
            "duplicate_records": duplicate_records,
            "duplicate_hive_timestamps": duplicate_hive_times,
            "timestamp_parse_failures": int(
                self.data[self.config.timestamp_column].isna().sum()
            ),
        }

    def _distribution_statistics(self) -> dict[str, Any]:
        numeric: dict[str, dict[str, float | int]] = {}
        for column in self.config.sensor_features:
            values = self.data[column].dropna()
            numeric[column] = {
                "n": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "median": values.median(),
                "q25": values.quantile(0.25),
                "q75": values.quantile(0.75),
                "min": values.min(),
                "max": values.max(),
                "skewness": values.skew(),
                "kurtosis": values.kurtosis(),
            }
        categorical: dict[str, Any] = {}
        for column in ("bee_stock", "apiary_context", "apiary_season"):
            if column in self.data:
                counts = self.data[column].fillna("Missing").astype(str).value_counts()
                categorical[column] = {
                    "unique_values": int(counts.size),
                    "value_counts": counts.to_dict(),
                }
        return {"numeric": numeric, "categorical": categorical}

    def _swarming_analysis(self) -> dict[str, Any]:
        distribution: dict[str, Any] = {}
        for key, column in (
            ("swarming_event", self.config.event_column),
            ("swarming_72h", self.config.target_column),
        ):
            valid = self.data[column].dropna()
            positive = int((valid == 1).sum())
            negative = int((valid == 0).sum())
            distribution[key] = {
                "count": {"positive": positive, "negative": negative},
                "rate": {
                    "positive": positive / len(valid) * 100 if len(valid) else 0.0,
                    "negative": negative / len(valid) * 100 if len(valid) else 0.0,
                },
            }

        by_hive = (
            self.data.groupby(self.config.hive_column, dropna=False)
            .agg(
                swarm_events=(self.config.event_column, "sum"),
                swarm_next_72h=(self.config.target_column, "sum"),
                total_records=(self.config.target_column, "count"),
            )
            .reset_index()
        )
        by_hive["swarm_rate"] = np.where(
            by_hive["total_records"] > 0,
            by_hive["swarm_events"] / by_hive["total_records"] * 100,
            np.nan,
        )
        by_hive["swarm_72h_rate"] = np.where(
            by_hive["total_records"] > 0,
            by_hive["swarm_next_72h"] / by_hive["total_records"] * 100,
            np.nan,
        )
        by_hive = by_hive.sort_values(
            ["swarm_rate", "swarm_events"], ascending=False
        )

        by_stock: list[dict[str, Any]] = []
        if "bee_stock" in self.data:
            stock = (
                self.data.groupby("bee_stock", dropna=False)
                .agg(
                    swarm_events=(self.config.event_column, "sum"),
                    total_records=(self.config.event_column, "count"),
                )
                .reset_index()
            )
            stock["swarm_rate"] = stock["swarm_events"] / stock["total_records"] * 100
            by_stock = stock.to_dict("records")

        return {
            "distribution": distribution,
            "by_hive": by_hive.to_dict("records"),
            "by_bee_stock": by_stock,
        }

    def _association_analysis(self) -> dict[str, Any]:
        target = self.data[self.config.target_column]
        rows = []
        for column in self.config.sensor_features:
            paired = self.data[[column, self.config.target_column]].dropna()
            negative = paired.loc[paired[self.config.target_column] == 0, column]
            positive = paired.loc[paired[self.config.target_column] == 1, column]
            rows.append(
                {
                    "feature": column,
                    "pearson": paired[column].corr(
                        paired[self.config.target_column], method="pearson"
                    ),
                    "spearman": paired[column].corr(
                        paired[self.config.target_column], method="spearman"
                    ),
                    "standardized_mean_difference": _standardized_mean_difference(
                        negative, positive
                    ),
                    "non_swarm_mean": negative.mean(),
                    "swarm_mean": positive.mean(),
                    "n_non_swarm": len(negative),
                    "n_swarm": len(positive),
                }
            )
        rows.sort(key=lambda row: abs(row["pearson"]) if pd.notna(row["pearson"]) else -1, reverse=True)

        matrix_columns = [*self.config.sensor_features, self.config.target_column]
        matrix = self.data[matrix_columns].corr(method="pearson")
        return {
            "feature_correlations": rows,
            "feature_matrix": matrix.to_dict(),
            "target_valid_records": int(target.notna().sum()),
        }

    def _temporal_analysis(self) -> dict[str, Any]:
        valid = self.data.dropna(subset=[self.config.timestamp_column]).copy()
        valid["hour"] = valid[self.config.timestamp_column].dt.hour
        valid["month"] = valid[self.config.timestamp_column].dt.to_period("M").astype(str)
        hourly = (
            valid.groupby("hour")[[*self.config.sensor_features, self.config.target_column]]
            .mean()
            .reset_index()
        )
        monthly = (
            valid.groupby("month")[[*self.config.sensor_features, self.config.target_column]]
            .mean()
            .reset_index()
        )
        return {
            "hourly": hourly.to_dict("records"),
            "monthly": monthly.to_dict("records"),
        }

    def _methodology(self) -> dict[str, Any]:
        return {
            "analysis_unit": "sensor record nested within hive",
            "primary_outcome": self.config.target_column,
            "event_indicator": self.config.event_column,
            "outlier_rule": "Tukey fences: Q1 − 1.5×IQR and Q3 + 1.5×IQR",
            "association_statistics": [
                "Pearson point-biserial correlation",
                "Spearman rank correlation",
                "Cohen standardized mean difference",
            ],
            "interpretation_note": (
                "EDA associations are descriptive and do not establish causality. "
                "Repeated records within hives reduce observation independence."
            ),
        }

    def _export_tables(self) -> None:
        reports = self.config.reports_dir
        pd.DataFrame(self.results["data_quality"]["missing_values"]).to_csv(
            reports / "missing_values.csv", index=False
        )
        pd.DataFrame.from_dict(
            self.results["data_quality"]["outliers"], orient="index"
        ).rename_axis("feature").reset_index().to_csv(
            reports / "outlier_summary.csv", index=False
        )
        pd.DataFrame.from_dict(
            self.results["distribution_stats"]["numeric"], orient="index"
        ).rename_axis("feature").reset_index().to_csv(
            reports / "sensor_descriptive_statistics.csv", index=False
        )
        pd.DataFrame(
            self.results["correlation_analysis"]["feature_correlations"]
        ).to_csv(reports / "swarming_association_statistics.csv", index=False)
        pd.DataFrame(self.results["swarming_analysis"]["by_hive"]).to_csv(
            reports / "swarming_by_hive.csv", index=False
        )
        pd.DataFrame(self.results["temporal_analysis"]["hourly"]).to_csv(
            reports / "hourly_patterns.csv", index=False
        )
        pd.DataFrame(self.results["temporal_analysis"]["monthly"]).to_csv(
            reports / "monthly_patterns.csv", index=False
        )

    def _create_figures(self) -> None:
        self._plot_feature_distributions()
        self._plot_swarm_indicators()
        self._plot_correlation_matrix()
        self._plot_associations()
        self._plot_temporal_patterns()
        self._plot_data_quality()

    def _save_figure(self, figure: plt.Figure, filename: str) -> None:
        figure.savefig(self.config.images_dir / filename, dpi=300)
        plt.close(figure)

    def _plot_feature_distributions(self) -> None:
        figure, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
        for axis, column in zip(axes.ravel(), self.config.sensor_features, strict=True):
            values = self.data[column].dropna()
            axis.hist(
                values,
                bins=45,
                color=self.NAVY,
                edgecolor="white",
                linewidth=0.25,
                alpha=0.9,
            )
            axis.axvline(values.median(), color=self.BURGUNDY, linewidth=1.2, linestyle="--")
            axis.set_title(self.config.sensor_labels[column], fontsize=9.5)
            axis.set_ylabel("Record count")
            axis.text(
                0.98,
                0.95,
                f"n = {len(values):,}\nMedian = {values.median():.2f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color=self.CHARCOAL,
            )
        figure.suptitle("Distribution of swarming-module sensor variables", fontsize=12)
        figure.supxlabel("Observed value; dashed line denotes median", fontsize=8.5)
        figure.tight_layout(rect=(0, 0.02, 1, 0.96))
        self._save_figure(figure, "feature_distribution.png")

    def _plot_swarm_indicators(self) -> None:
        figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
        valid = self.data[self.config.target_column].dropna()
        counts = valid.value_counts().reindex([0, 1], fill_value=0)
        bars = axes[0].bar(
            ["No swarm within 72 h", "Swarm within 72 h"],
            counts.values,
            color=[self.SLATE, self.BURGUNDY],
            width=0.62,
        )
        axes[0].set_title("Class distribution for the 72-hour target")
        axes[0].set_ylabel("Record count")
        axes[0].tick_params(axis="x", rotation=8)
        for bar, count in zip(bars, counts.values, strict=True):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count:,}\n({count / len(valid) * 100:.2f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        by_hive = pd.DataFrame(self.results["swarming_analysis"]["by_hive"]).head(12)
        by_hive = by_hive.sort_values("swarm_rate", ascending=True)
        axes[1].barh(
            by_hive[self.config.hive_column].astype(str),
            by_hive["swarm_rate"],
            color=self.NAVY,
            height=0.68,
        )
        axes[1].set_title("Swarm event rate by hive (highest 12)")
        axes[1].set_xlabel("Swarm-positive records (%)")
        axes[1].set_ylabel("Hive")
        figure.suptitle("Swarming outcome prevalence and between-hive heterogeneity", fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        self._save_figure(figure, "swarm_indicators.png")

    def _plot_correlation_matrix(self) -> None:
        columns = [*self.config.sensor_features, self.config.target_column]
        matrix = self.data[columns].corr(method="pearson")
        labels = [self.config.sensor_labels.get(column, "Swarm ≤72 h") for column in columns]
        figure, axis = plt.subplots(figsize=(8.8, 7.2))
        image = axis.imshow(matrix.values, vmin=-1, vmax=1, cmap="RdBu_r")
        axis.set_xticks(range(len(labels)), labels, rotation=38, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.grid(False)
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = matrix.iloc[row, column]
                text_color = "white" if abs(value) >= 0.55 else self.CHARCOAL
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)
        axis.set_title("Pearson correlation matrix for swarming sensor variables", pad=12)
        figure.colorbar(image, ax=axis, label="Correlation coefficient", shrink=0.82)
        figure.tight_layout()
        self._save_figure(figure, "correlation_matrix.png")

    def _plot_associations(self) -> None:
        rows = pd.DataFrame(self.results["correlation_analysis"]["feature_correlations"])
        rows["label"] = rows["feature"].map(self.config.sensor_labels)
        rows = rows.sort_values("pearson")
        positions = np.arange(len(rows))
        figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
        axes[0].barh(positions - 0.17, rows["pearson"], height=0.32, color=self.NAVY, label="Pearson")
        axes[0].barh(positions + 0.17, rows["spearman"], height=0.32, color=self.LIGHT_SLATE, label="Spearman")
        axes[0].axvline(0, color=self.CHARCOAL, linewidth=0.8)
        axes[0].set_yticks(positions, rows["label"])
        axes[0].set_xlabel("Correlation coefficient")
        axes[0].set_title("Sensor association with swarm ≤72 h")
        axes[0].legend(loc="lower right")

        colors = [self.BURGUNDY if value < 0 else self.NAVY for value in rows["standardized_mean_difference"]]
        axes[1].barh(positions, rows["standardized_mean_difference"], color=colors, height=0.55)
        axes[1].axvline(0, color=self.CHARCOAL, linewidth=0.8)
        axes[1].set_xlabel("Standardized mean difference (Cohen's d)")
        axes[1].set_title("Swarm vs non-swarm sensor separation")
        figure.suptitle("Magnitude and direction of sensor–swarming associations", fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        self._save_figure(figure, "top_correlations.png")

    def _plot_temporal_patterns(self) -> None:
        hourly = pd.DataFrame(self.results["temporal_analysis"]["hourly"])
        figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True)
        variables = (
            "internal_temperature_c",
            "internal_humidity_pct",
            "co2_ppm",
            "hive_weight_kg",
        )
        for axis, column in zip(axes.ravel(), variables, strict=True):
            axis.plot(hourly["hour"], hourly[column], color=self.NAVY, linewidth=1.6)
            axis.set_title(self.config.sensor_labels[column])
            axis.set_ylabel("Mean")
            axis.set_xticks(range(0, 24, 3))
        figure.suptitle("Mean diurnal profiles of internal hive measurements", fontsize=12)
        figure.supxlabel("Hour of day")
        figure.tight_layout(rect=(0.02, 0.03, 1, 0.95))
        self._save_figure(figure, "temporal_patterns.png")

    def _plot_data_quality(self) -> None:
        missing = pd.DataFrame(self.results["data_quality"]["missing_values"])
        outliers = pd.DataFrame.from_dict(
            self.results["data_quality"]["outliers"], orient="index"
        ).rename_axis("feature").reset_index()
        figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        missing = missing.loc[missing["column"].isin(self.config.sensor_features)]
        missing["label"] = missing["column"].map(self.config.sensor_labels)
        axes[0].barh(missing["label"], missing["missing_percentage"], color=self.SLATE)
        axes[0].set_title("Missing observations by sensor")
        axes[0].set_xlabel("Missing records (%)")
        outliers["label"] = outliers["feature"].map(self.config.sensor_labels)
        axes[1].barh(outliers["label"], outliers["percentage"], color=self.NAVY)
        axes[1].set_title("Observations outside Tukey fences")
        axes[1].set_xlabel("Flagged records (%)")
        figure.suptitle("Data-quality diagnostics for swarming predictors", fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        self._save_figure(figure, "data_quality.png")

    def _write_dashboard(self) -> None:
        initial = self.results["initial_stats"]
        rate = self.results["swarming_analysis"]["distribution"]["swarming_72h"]["rate"]["positive"]
        correlations = self.results["correlation_analysis"]["feature_correlations"]
        dashboard = {
            "summary": {
                "total_records": initial["total_records"],
                "total_hives": initial["total_hives"],
                "time_days": initial["time_range"]["days"],
                "time_start": initial["time_range"]["start"],
                "time_end": initial["time_range"]["end"],
                "swarm_rate": rate,
            },
            "data_quality": self.results["data_quality"],
            "distribution_stats": self.results["distribution_stats"],
            "swarming_analysis": self.results["swarming_analysis"],
            "correlation_analysis": {
                "top_features": [
                    {
                        "feature": row["feature"],
                        "correlation": row["pearson"],
                        "spearman": row["spearman"],
                        "effect_size": row["standardized_mean_difference"],
                    }
                    for row in correlations
                ]
            },
            "methodology": self.results["methodology"],
            "generated_figures": [
                "feature_distribution.png",
                "swarm_indicators.png",
                "correlation_matrix.png",
                "top_correlations.png",
                "temporal_patterns.png",
                "data_quality.png",
            ],
        }
        (self.config.output_dir / "dashboard.json").write_text(
            json.dumps(_json_safe(dashboard), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _write_report(self) -> None:
        overview = self.results["initial_stats"]
        quality = self.results["data_quality"]
        prevalence = self.results["swarming_analysis"]["distribution"]["swarming_72h"]
        associations = self.results["correlation_analysis"]["feature_correlations"]
        lines = [
            "SWARMING MODULE — EXPLORATORY DATA ANALYSIS REPORT",
            "=" * 58,
            "",
            "Scope",
            "-----",
            f"Records: {overview['total_records']:,}",
            f"Hives: {overview['total_hives']}",
            f"Observation period: {overview['time_range']['start']} to {overview['time_range']['end']}",
            f"Coverage: {overview['time_range']['days']:,} days",
            "",
            "Outcome prevalence",
            "------------------",
            f"Positive 72-hour target records: {prevalence['count']['positive']:,}",
            f"Positive 72-hour target rate: {prevalence['rate']['positive']:.4f}%",
            "",
            "Data quality",
            "------------",
            f"Exact duplicate records: {quality['duplicate_records']:,}",
            f"Duplicate hive–timestamp rows: {quality['duplicate_hive_timestamps']:,}",
            f"Unparseable timestamps: {quality['timestamp_parse_failures']:,}",
            "",
            "Sensor associations with swarming within 72 hours",
            "--------------------------------------------------",
        ]
        for row in associations:
            lines.append(
                f"{self.config.sensor_labels[row['feature']]}: "
                f"Pearson={row['pearson']:.4f}, Spearman={row['spearman']:.4f}, "
                f"Cohen d={row['standardized_mean_difference']:.4f}"
            )
        lines.extend(
            [
                "",
                "Interpretation constraint",
                "-------------------------",
                self.results["methodology"]["interpretation_note"],
            ]
        )
        (self.config.reports_dir / "feature_analysis_summary.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def run_swarming_eda(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Public entry point used by scripts and tests."""

    config = EDAConfig.default()
    if data_path is not None or output_dir is not None:
        config = EDAConfig(
            backend_dir=config.backend_dir,
            data_path=Path(data_path) if data_path is not None else config.data_path,
            output_dir=Path(output_dir) if output_dir is not None else config.output_dir,
        )
    return SwarmingEDA(config).run()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_swarming_eda()
    summary = results["initial_stats"]
    print("\nSwarming EDA completed successfully")
    print(f"Records analysed: {summary['total_records']:,}")
    print(f"Hives analysed: {summary['total_hives']}")
    print(f"Outputs: {EDAConfig.default().output_dir}")


if __name__ == "__main__":
    main()