from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .schema import HIVE_COLUMN, SENSOR_COLUMNS, TARGET_COLUMNS, TIMESTAMP_COLUMN

SENSOR_LABELS = {
    "temperature_c": "Temperature (°C)",
    "humidity_pct": "Humidity (%)",
    "co2_ppm": "CO₂ (ppm)",
    "weight_kg": "Hive weight (kg)",
}

TARGET_LABELS = {
    "brood_health_healthy_1": "Brood healthy",
    "swarming_happened_1": "Swarming",
    "absconding_happened_1": "Absconding",
    "honey_harvested_1": "Harvesting",
}


def generate_common_eda(df: pd.DataFrame, output_directory: str | Path) -> None:
    """Generate common tabular reports and presentation-ready figures.

    The generated PNG files are served directly by the backend. The frontend therefore
    shows the exact same figures that are saved under artifacts/reports/common_eda.
    """

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    data = df.copy()
    data[TIMESTAMP_COLUMN] = pd.to_datetime(data[TIMESTAMP_COLUMN], errors="coerce")

    numeric_summary = data[list(SENSOR_COLUMNS)].describe().T
    numeric_summary.to_csv(output / "sensor_summary.csv")

    missing = data[list(SENSOR_COLUMNS) + list(TARGET_COLUMNS)].isna().sum().rename("missing")
    missing.to_csv(output / "missing_values.csv")

    hive_summary = (
        data.groupby(HIVE_COLUMN)
        .agg(
            rows=(TIMESTAMP_COLUMN, "size"),
            start=(TIMESTAMP_COLUMN, "min"),
            end=(TIMESTAMP_COLUMN, "max"),
        )
        .reset_index()
    )
    hive_summary.to_csv(output / "hive_coverage.csv", index=False)

    target_summary = {
        target: data[target].value_counts(dropna=False).sort_index().to_dict()
        for target in TARGET_COLUMNS
    }
    (output / "target_balance.json").write_text(
        json.dumps(target_summary, indent=2, default=str), encoding="utf-8"
    )

    correlation = data[list(SENSOR_COLUMNS)].corr()
    correlation.to_csv(output / "sensor_correlation.csv")

    temporal = data.dropna(subset=[TIMESTAMP_COLUMN]).copy()
    temporal["hour"] = temporal[TIMESTAMP_COLUMN].dt.hour
    temporal["weekday"] = temporal[TIMESTAMP_COLUMN].dt.day_name()
    temporal["month"] = temporal[TIMESTAMP_COLUMN].dt.to_period("M").astype(str)

    hourly = temporal.groupby("hour")[list(SENSOR_COLUMNS)].mean().reset_index()
    weekday = temporal.groupby("weekday")[list(SENSOR_COLUMNS)].mean().reset_index()
    monthly = temporal.groupby("month")[list(SENSOR_COLUMNS)].mean().reset_index()
    monthly_targets = temporal.groupby("month")[list(TARGET_COLUMNS)].sum().reset_index()
    target_by_hive = data.groupby(HIVE_COLUMN)[list(TARGET_COLUMNS)].sum().reset_index()

    hourly.to_csv(output / "hourly_patterns.csv", index=False)
    weekday.to_csv(output / "weekday_patterns.csv", index=False)
    monthly.to_csv(output / "monthly_patterns.csv", index=False)
    monthly_targets.to_csv(output / "monthly_target_counts.csv", index=False)
    target_by_hive.to_csv(output / "target_by_hive.csv", index=False)

    outlier_rows = []
    for column in SENSOR_COLUMNS:
        series = pd.to_numeric(data[column], errors="coerce").dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        outlier_rows.append(
            {
                "sensor": column,
                "count": count,
                "percentage": (count / len(series) * 100) if len(series) else 0.0,
                "lower_bound": lower,
                "upper_bound": upper,
            }
        )
    pd.DataFrame(outlier_rows).to_csv(output / "outlier_summary.csv", index=False)

    _plot_target_balance(data, output / "target_balance.png")
    _plot_sensor_distributions(data, output / "sensor_distributions.png")
    _plot_hive_row_counts(data, output / "rows_per_hive.png")
    _plot_correlation_heatmap(correlation, output / "correlation_heatmap.png")
    _plot_hourly_patterns(hourly, output / "hourly_sensor_patterns.png")
    _plot_monthly_patterns(monthly, output / "monthly_sensor_trends.png")
    _plot_outlier_percentages(outlier_rows, output / "outlier_percentages.png")
    _plot_target_positive_rates(data, output / "target_positive_rates_log.png")
    _plot_monthly_event_timeline(monthly_targets, output / "monthly_event_timeline.png")
    _plot_target_cooccurrence(data, output / "target_cooccurrence.png")


def _plot_target_balance(df: pd.DataFrame, path: Path) -> None:
    positive_counts = [int(df[column].sum()) for column in TARGET_COLUMNS]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar([TARGET_LABELS[column] for column in TARGET_COLUMNS], positive_counts)
    axis.set_title("Positive labels by target")
    axis.set_ylabel("Positive records")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_sensor_distributions(df: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, column in zip(axes.ravel(), SENSOR_COLUMNS, strict=True):
        axis.hist(df[column].dropna(), bins=50)
        axis.set_title(SENSOR_LABELS[column])
        axis.set_ylabel("Records")
    figure.suptitle("Sensor-value distributions", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_hive_row_counts(df: pd.DataFrame, path: Path) -> None:
    counts = df[HIVE_COLUMN].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(counts.index.astype(str), counts.values)
    axis.set_title("Records per hive")
    axis.set_ylabel("Rows")
    axis.tick_params(axis="x", rotation=90)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_correlation_heatmap(correlation: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(correlation.values, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [SENSOR_LABELS[column].split(" (")[0] for column in correlation.columns]
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Sensor correlation matrix")
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_hourly_patterns(hourly: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, column in zip(axes.ravel(), SENSOR_COLUMNS, strict=True):
        axis.plot(hourly["hour"], hourly[column])
        axis.set_title(SENSOR_LABELS[column])
        axis.set_xlabel("Hour of day")
        axis.grid(alpha=0.25)
    figure.suptitle("Hourly sensor patterns")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_monthly_patterns(monthly: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    positions = np.arange(len(monthly))
    for axis, column in zip(axes.ravel(), SENSOR_COLUMNS, strict=True):
        axis.plot(positions, monthly[column])
        axis.set_title(SENSOR_LABELS[column])
        axis.grid(alpha=0.25)
    tick_step = max(1, len(monthly) // 10)
    ticks = positions[::tick_step]
    labels = monthly["month"].iloc[::tick_step]
    for axis in axes[-1]:
        axis.set_xticks(ticks, labels, rotation=35, ha="right")
    figure.suptitle("Monthly sensor trends")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_outlier_percentages(rows: list[dict[str, object]], path: Path) -> None:
    labels = [SENSOR_LABELS[str(row["sensor"])].split(" (")[0] for row in rows]
    values = [float(row["percentage"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, values)
    axis.set_title("IQR outlier percentage by sensor")
    axis.set_ylabel("Outliers (%)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_target_positive_rates(df: pd.DataFrame, path: Path) -> None:
    total = len(df)
    rates = [max((int(df[column].sum()) / total) * 100, 1e-8) for column in TARGET_COLUMNS]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar([TARGET_LABELS[column] for column in TARGET_COLUMNS], rates)
    axis.set_yscale("log")
    axis.set_title("Positive-label rates on a logarithmic scale")
    axis.set_ylabel("Positive records (%) — log scale")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_monthly_event_timeline(monthly_targets: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    positions = np.arange(len(monthly_targets))
    for axis, target in zip(axes.ravel(), TARGET_COLUMNS, strict=True):
        axis.plot(positions, monthly_targets[target], marker="o", markersize=2)
        axis.set_title(TARGET_LABELS[target])
        axis.set_ylabel("Positive records")
        axis.grid(alpha=0.25)
    tick_step = max(1, len(monthly_targets) // 10)
    ticks = positions[::tick_step]
    labels = monthly_targets["month"].iloc[::tick_step]
    for axis in axes[-1]:
        axis.set_xticks(ticks, labels, rotation=35, ha="right")
    figure.suptitle("Monthly target-event timeline")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_target_cooccurrence(df: pd.DataFrame, path: Path) -> None:
    matrix = np.zeros((len(TARGET_COLUMNS), len(TARGET_COLUMNS)), dtype=int)
    for row_index, row_target in enumerate(TARGET_COLUMNS):
        for column_index, column_target in enumerate(TARGET_COLUMNS):
            matrix[row_index, column_index] = int(
                ((df[row_target] == 1) & (df[column_target] == 1)).sum()
            )

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    labels = [TARGET_LABELS[column] for column in TARGET_COLUMNS]
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Target-label co-occurrence")
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Co-occurring records")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
