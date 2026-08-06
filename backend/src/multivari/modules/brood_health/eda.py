from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from .analyzer import HEALTH_LEVELS, compute_condition_history
from .config import PATHS
from .features import SENSORS, TARGET_COLUMN, normalise_historical
from .scoring import BroodHealthScoreConfig, score_definition

SENSOR_META = {
    "temperature_c": {"label": "Temperature", "unit": "°C"},
    "humidity_pct": {"label": "Humidity", "unit": "%"},
    "co2_ppm": {"label": "CO₂", "unit": "ppm"},
    "weight_kg": {"label": "Hive weight", "unit": "kg"},
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in frame[columns].to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                clean[key] = value.isoformat()
            elif isinstance(value, (np.integer, int)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                clean[key] = _finite(value)
            else:
                clean[key] = value
        output.append(clean)
    return output


def _load_source(path: Path | None = None) -> pd.DataFrame:
    source = Path(path or PATHS.clean_data)
    if source.exists():
        if source.suffix.lower() in {".xlsx", ".xls"}:
            return normalise_historical(pd.read_excel(source, sheet_name="Common_Dataset"))
        if source.suffix.lower() == ".csv":
            return normalise_historical(pd.read_csv(source))
        try:
            return normalise_historical(pd.read_parquet(source))
        except ImportError:
            # A local environment may not have a Parquet engine even though the raw
            # workbook is available. Falling back keeps EDA reproducible without
            # silently changing the brood-specific preprocessing logic.
            pass
    if PATHS.raw_workbook.exists():
        return normalise_historical(pd.read_excel(PATHS.raw_workbook, sheet_name="Common_Dataset"))
    raise FileNotFoundError(
        "The cleaned common dataset is missing. Run scripts/run_common_pipeline.py before requesting brood-health EDA."
    )


def _cohens_d(healthy: pd.Series, unhealthy: pd.Series) -> float | None:
    a = pd.to_numeric(healthy, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(unhealthy, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return _finite((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _sensor_statistics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    healthy = frame[TARGET_COLUMN].eq(1)
    for sensor in SENSORS:
        values = pd.to_numeric(frame[sensor], errors="coerce")
        healthy_values = values[healthy]
        unhealthy_values = values[~healthy]
        rows.append(
            {
                "sensor": sensor,
                **SENSOR_META[sensor],
                "overall_mean": _finite(values.mean()),
                "overall_std": _finite(values.std(ddof=0)),
                "overall_median": _finite(values.median()),
                "overall_min": _finite(values.min()),
                "overall_max": _finite(values.max()),
                "healthy_mean": _finite(healthy_values.mean()),
                "healthy_std": _finite(healthy_values.std(ddof=0)),
                "healthy_median": _finite(healthy_values.median()),
                "unhealthy_mean": _finite(unhealthy_values.mean()),
                "unhealthy_std": _finite(unhealthy_values.std(ddof=0)),
                "unhealthy_median": _finite(unhealthy_values.median()),
                "mean_difference": _finite(healthy_values.mean() - unhealthy_values.mean()),
                "cohens_d": _cohens_d(healthy_values, unhealthy_values),
                "target_correlation": _finite(values.corr(frame[TARGET_COLUMN])),
                "missing": int(values.isna().sum()),
            }
        )
    return rows


def _sensor_distributions(frame: pd.DataFrame, bins: int = 28) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for sensor in SENSORS:
        values = pd.to_numeric(frame[sensor], errors="coerce").dropna()
        lower = float(values.quantile(0.005))
        upper = float(values.quantile(0.995))
        if lower >= upper:
            lower, upper = float(values.min()), float(values.max())
        edges = np.linspace(lower, upper, bins + 1)
        rows: list[dict[str, Any]] = []
        for label, name in ((1, "healthy"), (0, "unhealthy")):
            selected = pd.to_numeric(frame.loc[frame[TARGET_COLUMN].eq(label), sensor], errors="coerce").dropna()
            counts, _ = np.histogram(selected.clip(lower, upper), bins=edges)
            denominator = max(int(counts.sum()), 1)
            for index, count in enumerate(counts):
                if len(rows) <= index:
                    rows.append(
                        {
                            "bin_start": float(edges[index]),
                            "bin_end": float(edges[index + 1]),
                            "bin_mid": float((edges[index] + edges[index + 1]) / 2.0),
                        }
                    )
                rows[index][name] = int(count)
                rows[index][f"{name}_percentage"] = float(count / denominator * 100.0)
        output[sensor] = rows
    return output


def _temporal_profiles(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    data = frame.copy()
    data["hour"] = data["timestamp"].dt.hour
    data["weekday_number"] = data["timestamp"].dt.dayofweek
    data["weekday"] = data["timestamp"].dt.day_name().str[:3]
    month_source = data["timestamp"]
    if month_source.dt.tz is not None:
        month_source = month_source.dt.tz_convert("UTC").dt.tz_localize(None)
    data["month"] = month_source.dt.to_period("M").astype(str)

    def aggregate(group_columns: list[str]) -> pd.DataFrame:
        return (
            data.groupby(group_columns, observed=True)
            .agg(
                healthy_rate=(TARGET_COLUMN, "mean"),
                records=(TARGET_COLUMN, "size"),
                unhealthy_count=(TARGET_COLUMN, lambda values: int((values == 0).sum())),
                condition_score=("condition_score", "mean"),
                temperature=("temperature_c", "mean"),
                humidity=("humidity_pct", "mean"),
                co2=("co2_ppm", "mean"),
                weight=("weight_kg", "mean"),
            )
            .reset_index()
        )

    hourly = aggregate(["hour"])
    weekday = aggregate(["weekday_number", "weekday"]).sort_values("weekday_number")
    monthly = aggregate(["month"])
    for table in (hourly, weekday, monthly):
        table["healthy_rate"] *= 100.0
    return {
        "hourly": _records(hourly, list(hourly.columns)),
        "weekday": _records(weekday, list(weekday.columns)),
        "monthly": _records(monthly, list(monthly.columns)),
    }


def _hive_profiles(frame: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        frame.groupby("hive_id", observed=True)
        .agg(
            records=(TARGET_COLUMN, "size"),
            healthy_rate=(TARGET_COLUMN, "mean"),
            unhealthy_count=(TARGET_COLUMN, lambda values: int((values == 0).sum())),
            condition_score=("condition_score", "mean"),
            temperature=("temperature_c", "mean"),
            humidity=("humidity_pct", "mean"),
            co2=("co2_ppm", "mean"),
            weight=("weight_kg", "mean"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
        )
        .reset_index()
    )
    grouped["healthy_rate"] *= 100.0
    grouped["unhealthy_rate"] = 100.0 - grouped["healthy_rate"]
    return _records(grouped, list(grouped.columns))


def _transitions_and_episodes(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values(["hive_id", "timestamp"]).copy()
    previous = ordered.groupby("hive_id", sort=False)[TARGET_COLUMN].shift(1)
    valid = previous.notna()
    matrix = pd.crosstab(previous[valid].astype(int), ordered.loc[valid, TARGET_COLUMN].astype(int))
    for index in (0, 1):
        if index not in matrix.index:
            matrix.loc[index] = 0
        if index not in matrix.columns:
            matrix[index] = 0
    matrix = matrix.sort_index().sort_index(axis=1)
    probability = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    changed = ordered[TARGET_COLUMN].ne(previous) | ordered["hive_id"].ne(ordered["hive_id"].shift(1))
    run_id = changed.astype("int64").cumsum()
    episodes = (
        ordered.assign(run_id=run_id)
        .groupby(["hive_id", "run_id", TARGET_COLUMN], observed=True)
        .agg(start=("timestamp", "min"), end=("timestamp", "max"), duration_hours=(TARGET_COLUMN, "size"))
        .reset_index()
    )
    episode_summary = (
        episodes.groupby(TARGET_COLUMN)["duration_hours"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={TARGET_COLUMN: "status"})
    )
    episode_summary["status"] = episode_summary["status"].map({0: "Unhealthy", 1: "Healthy"})
    onset_count = int(((previous == 1) & (ordered[TARGET_COLUMN] == 0)).sum())
    recovery_count = int(((previous == 0) & (ordered[TARGET_COLUMN] == 1)).sum())
    return {
        "counts": [
            {"from": "Unhealthy", "to": "Unhealthy", "count": int(matrix.loc[0, 0]), "probability": float(probability.loc[0, 0] * 100)},
            {"from": "Unhealthy", "to": "Healthy", "count": int(matrix.loc[0, 1]), "probability": float(probability.loc[0, 1] * 100)},
            {"from": "Healthy", "to": "Unhealthy", "count": int(matrix.loc[1, 0]), "probability": float(probability.loc[1, 0] * 100)},
            {"from": "Healthy", "to": "Healthy", "count": int(matrix.loc[1, 1]), "probability": float(probability.loc[1, 1] * 100)},
        ],
        "unhealthy_onsets": onset_count,
        "recoveries": recovery_count,
        "episode_summary": _records(episode_summary, list(episode_summary.columns)),
    }


def _precursor_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values(["hive_id", "timestamp"]).copy()
    previous = ordered.groupby("hive_id", sort=False)[TARGET_COLUMN].shift(1)
    onset_positions = np.flatnonzero(((previous == 1) & (ordered[TARGET_COLUMN] == 0)).to_numpy())
    windows = ((0, 6, "0–6 h before"), (6, 12, "6–12 h before"), (12, 24, "12–24 h before"), (24, 48, "24–48 h before"))
    aggregate: dict[tuple[str, str], list[float]] = {}
    baseline: dict[str, list[float]] = {sensor: [] for sensor in SENSORS}
    accepted_events = 0

    for position in onset_positions:
        hive_id = ordered.iloc[position]["hive_id"]
        hive_indices = ordered.index[ordered["hive_id"].eq(hive_id)].to_numpy()
        local_position_candidates = np.flatnonzero(hive_indices == ordered.index[position])
        if not len(local_position_candidates):
            continue
        local_position = int(local_position_candidates[0])
        if local_position < 96:
            continue
        hive = ordered.loc[hive_indices].reset_index(drop=True)
        accepted_events += 1
        for sensor in SENSORS:
            base_value = pd.to_numeric(hive.loc[local_position - 96 : local_position - 49, sensor], errors="coerce").mean()
            baseline[sensor].append(float(base_value))
            for start, end, label in windows:
                values = pd.to_numeric(hive.loc[local_position - end : local_position - start - 1, sensor], errors="coerce")
                aggregate.setdefault((sensor, label), []).append(float(values.mean()))

    rows: list[dict[str, Any]] = []
    for sensor in SENSORS:
        base = float(np.nanmean(baseline[sensor])) if baseline[sensor] else float("nan")
        for _, _, label in windows:
            values = aggregate.get((sensor, label), [])
            mean = float(np.nanmean(values)) if values else float("nan")
            rows.append(
                {
                    "sensor": sensor,
                    **SENSOR_META[sensor],
                    "window": label,
                    "mean": _finite(mean),
                    "baseline_mean": _finite(base),
                    "delta_from_baseline": _finite(mean - base),
                    "events": len(values),
                }
            )
    return {"accepted_onsets": accepted_events, "baseline_window": "48–96 h before onset", "rows": rows}


def _data_quality(frame: pd.DataFrame) -> dict[str, Any]:
    missing = [{"column": column, "count": int(frame[column].isna().sum()), "percentage": float(frame[column].isna().mean() * 100)} for column in frame.columns]
    outliers: list[dict[str, Any]] = []
    for sensor in SENSORS:
        values = pd.to_numeric(frame[sensor], errors="coerce")
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (values < lower) | (values > upper)
        outliers.append(
            {
                "sensor": sensor,
                **SENSOR_META[sensor],
                "count": int(mask.sum()),
                "percentage": float(mask.mean() * 100),
                "lower_bound": _finite(lower),
                "upper_bound": _finite(upper),
            }
        )
    duplicate_count = int(frame.duplicated(["hive_id", "timestamp"]).sum())
    gaps = []
    for hive_id, group in frame.groupby("hive_id", sort=False):
        delta = group.sort_values("timestamp")["timestamp"].diff().dt.total_seconds().div(3600)
        gaps.append(int((delta > 1.5).sum()))
    return {
        "missing": missing,
        "total_missing": int(frame.isna().sum().sum()),
        "duplicate_hive_timestamps": duplicate_count,
        "detected_time_gaps": int(sum(gaps)),
        "outliers": outliers,
    }


def _correlation(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [*SENSORS, "condition_score", TARGET_COLUMN]
    correlation = frame[columns].corr(numeric_only=True)
    return [
        {"row": row, "column": column, "value": _finite(correlation.loc[row, column])}
        for row in columns
        for column in columns
    ]


def _scatter_sample(frame: pd.DataFrame, maximum: int = 2000) -> list[dict[str, Any]]:
    if len(frame) > maximum:
        positions = np.linspace(0, len(frame) - 1, maximum, dtype=int)
        sample = frame.iloc[positions]
    else:
        sample = frame
    columns = ["hive_id", "timestamp", *SENSORS, TARGET_COLUMN, "condition_score"]
    return _records(sample, columns)


def _condition_level_balance(frame: pd.DataFrame) -> list[dict[str, Any]]:
    counts = frame["condition_level"].value_counts().reindex([item["level"] for item in HEALTH_LEVELS], fill_value=0)
    total = max(int(counts.sum()), 1)
    return [{"level": level, "count": int(count), "percentage": float(count / total * 100)} for level, count in counts.items()]


def _save_report_images(payload: dict[str, Any], frame: pd.DataFrame, directory: Path) -> list[dict[str, str]]:
    directory.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, str]] = []

    def save(fig: plt.Figure, filename: str, title: str) -> None:
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        images.append({"filename": filename, "title": title, "url": f"/api/brood-health/reports/{filename}"})

    balance = payload["class_balance"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([item["label"] for item in balance], [item["count"] for item in balance])
    ax.set_ylabel("Records")
    ax.set_title("Observed brood-health target balance")
    save(fig, "observed_target_balance.png", "Observed target balance")

    statistics = payload["sensor_statistics"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(statistics))
    width = 0.36
    ax.bar(x - width / 2, [item["healthy_mean"] for item in statistics], width, label="Healthy")
    ax.bar(x + width / 2, [item["unhealthy_mean"] for item in statistics], width, label="Unhealthy")
    ax.set_xticks(x, [item["label"] for item in statistics])
    ax.set_title("Mean sensor values by observed brood-health status")
    ax.legend()
    save(fig, "sensor_means_by_status.png", "Sensor means by observed status")

    hourly = pd.DataFrame(payload["temporal"]["hourly"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hourly["hour"], hourly["healthy_rate"], marker="o")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Healthy records (%)")
    ax.set_title("Hourly brood-health target pattern")
    save(fig, "hourly_healthy_rate.png", "Hourly healthy-rate pattern")

    hives = pd.DataFrame(payload["hive_profiles"]).sort_values("healthy_rate")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(hives["hive_id"], hives["healthy_rate"])
    ax.set_xlabel("Healthy records (%)")
    ax.set_title("Observed healthy rate by hive")
    save(fig, "healthy_rate_by_hive.png", "Healthy rate by hive")

    transition = payload["transitions"]["counts"]
    matrix = np.array([[transition[0]["probability"], transition[1]["probability"]], [transition[2]["probability"], transition[3]["probability"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, vmin=0, vmax=100)
    ax.set_xticks([0, 1], ["Unhealthy", "Healthy"])
    ax.set_yticks([0, 1], ["Unhealthy", "Healthy"])
    ax.set_xlabel("Next hour")
    ax.set_ylabel("Current hour")
    ax.set_title("One-hour status transition probabilities")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:.2f}%", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Probability (%)")
    save(fig, "status_transition_matrix.png", "Status transition matrix")

    precursor = pd.DataFrame(payload["precursor_analysis"]["rows"])
    fig, ax = plt.subplots(figsize=(10, 5))
    for sensor, group in precursor.groupby("sensor", sort=False):
        ax.plot(group["window"], group["delta_from_baseline"], marker="o", label=SENSOR_META[sensor]["label"])
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Change from 48–96 h baseline")
    ax.set_title("Sensor changes before unhealthy-status onset")
    ax.legend()
    save(fig, "unhealthy_onset_precursors.png", "Precursor changes before unhealthy onset")

    return images


def _active_score_config() -> BroodHealthScoreConfig:
    """Use calibrated weights when the v4 model exists; otherwise use the prior."""

    if PATHS.model_bundle.exists():
        try:
            bundle = joblib.load(PATHS.model_bundle)
            return BroodHealthScoreConfig.from_dict(bundle.get("score_config"))
        except Exception:
            pass
    return BroodHealthScoreConfig()


def _component_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = (
        ("temperature_component", "Temperature suitability"),
        ("humidity_component", "Humidity suitability"),
        ("co2_component", "CO₂ suitability"),
        ("weight_component", "Relative weight stability"),
    )
    return [
        {
            "component": column,
            "label": label,
            "mean": _finite(frame[column].mean()),
            "std": _finite(frame[column].std(ddof=0)),
            "median": _finite(frame[column].median()),
            "minimum": _finite(frame[column].min()),
            "maximum": _finite(frame[column].max()),
        }
        for column, label in columns
    ]


def build_brood_eda(*, data_path: Path | None = None, save_cache: bool = True) -> dict[str, Any]:
    frame = _load_source(data_path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"The brood-health EDA source does not contain {TARGET_COLUMN}")
    frame = frame.dropna(subset=[TARGET_COLUMN]).copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    score_config = _active_score_config()
    frame = compute_condition_history(frame, score_config=score_config)

    healthy_count = int(frame[TARGET_COLUMN].sum())
    total = len(frame)
    unhealthy_count = total - healthy_count
    payload: dict[str, Any] = {
        "meta": {
            "source": str(Path(data_path or PATHS.clean_data).name),
            "target": TARGET_COLUMN,
            "target_kind": "observed binary label",
            "records": total,
            "hives": int(frame["hive_id"].nunique()),
            "analysis_start": frame["timestamp"].min().isoformat(),
            "analysis_end": frame["timestamp"].max().isoformat(),
            "sampling_frequency": "1 hour",
            "healthy_count": healthy_count,
            "unhealthy_count": unhealthy_count,
            "healthy_rate": float(healthy_count / max(total, 1) * 100.0),
            "unhealthy_rate": float(unhealthy_count / max(total, 1) * 100.0),
            "condition_score_note": "The 1–100 condition score is a transparent sensor-derived research index. The historical binary status is used for EDA and training-only weight calibration, never as a forecasting feature.",
        },
        "class_balance": [
            {"label": "Healthy", "target_value": 1, "count": healthy_count, "percentage": float(healthy_count / max(total, 1) * 100.0)},
            {"label": "Unhealthy", "target_value": 0, "count": unhealthy_count, "percentage": float(unhealthy_count / max(total, 1) * 100.0)},
        ],
        "sensor_statistics": _sensor_statistics(frame),
        "sensor_distributions": _sensor_distributions(frame),
        "temporal": _temporal_profiles(frame),
        "hive_profiles": _hive_profiles(frame),
        "transitions": _transitions_and_episodes(frame),
        "precursor_analysis": _precursor_analysis(frame),
        "data_quality": _data_quality(frame),
        "correlation": _correlation(frame),
        "condition_level_balance": _condition_level_balance(frame),
        "score_component_summary": _component_summary(frame),
        "score_definition": score_definition(score_config),
        "scatter_sample": _scatter_sample(frame),
        "health_level_definitions": list(HEALTH_LEVELS),
        "methodology": {
            "split_rule": "Complete-hive 60/20/20 train, validation and test holdout. Score weights are calibrated on training hives only.",
            "forecast_horizon_hours": 6,
            "leakage_controls": [
                "No random row shuffle.",
                "Features use current and past sensor observations only.",
                "The primary model target is the exact score at +6 hours; the predicted minimum inside the trajectory is secondary.",
                "Target labels are never used as model input features.",
            ],
        },
    }
    payload["generated_images"] = _save_report_images(payload, frame, PATHS.report_dir / "eda")
    if save_cache:
        PATHS.metrics_dir.mkdir(parents=True, exist_ok=True)
        PATHS.eda_cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload