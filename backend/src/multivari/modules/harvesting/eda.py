from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu

HIVE_COLUMN = "hive_id"
TIMESTAMP_COLUMN = "timestamp"
SENSOR_COLUMNS = ["weight_kg", "temperature_c", "humidity_pct", "co2_ppm"]
RELATIONSHIP_FEATURES = [
    "weight_change_24h",
    "weight_change_72h",
    "weight_change_168h",
    "distance_from_max_168h",
    "relative_to_max_168h",
    "weight_std_24h",
    "temperature_mean_24h",
    "temperature_std_24h",
    "temperature_change_24h",
    "humidity_mean_24h",
    "humidity_std_24h",
    "humidity_change_24h",
    "co2_mean_24h",
    "co2_std_24h",
    "co2_change_24h",
    "environmental_stability_24h",
]


def _resolve_path(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _require_columns(frame: pd.DataFrame, required: set[str], *, frame_name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _safe_filename(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return cleaned.strip("_") or "event"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if math.isnan(float(value)) else float(value)
    if pd.isna(value):
        return None
    return value


def add_research_eda_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create past-only descriptive features for harvest EDA."""
    _require_columns(
        frame,
        {HIVE_COLUMN, TIMESTAMP_COLUMN, *SENSOR_COLUMNS},
        frame_name="Common cleaned dataset",
    )
    result = frame.copy()
    result[TIMESTAMP_COLUMN] = pd.to_datetime(result[TIMESTAMP_COLUMN], errors="raise")
    result = result.sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)
    grouped = result.groupby(HIVE_COLUMN, sort=False)

    for hours in (1, 6, 24, 72, 168):
        result[f"weight_change_{hours}h"] = grouped["weight_kg"].diff(hours)

    result["temperature_change_24h"] = grouped["temperature_c"].diff(24)
    result["humidity_change_24h"] = grouped["humidity_pct"].diff(24)
    result["co2_change_24h"] = grouped["co2_ppm"].diff(24)

    rolling_map = {
        "weight_kg": "weight",
        "temperature_c": "temperature",
        "humidity_pct": "humidity",
        "co2_ppm": "co2",
    }
    for source, prefix in rolling_map.items():
        result[f"{prefix}_mean_24h"] = grouped[source].transform(
            lambda series: series.rolling(24, min_periods=12).mean()
        )
        result[f"{prefix}_std_24h"] = grouped[source].transform(
            lambda series: series.rolling(24, min_periods=12).std()
        )

    result["recent_max_weight_168h"] = grouped["weight_kg"].transform(
        lambda series: series.rolling(168, min_periods=24).max()
    )
    result["distance_from_max_168h"] = result["recent_max_weight_168h"] - result["weight_kg"]
    result["relative_to_max_168h"] = np.where(
        result["recent_max_weight_168h"].gt(0),
        result["weight_kg"] / result["recent_max_weight_168h"],
        np.nan,
    )

    stability_components = pd.concat(
        [
            result["temperature_std_24h"],
            result["humidity_std_24h"] / 10.0,
            result["co2_std_24h"] / 1000.0,
        ],
        axis=1,
    )
    result["environmental_stability_24h"] = 1.0 / (1.0 + stability_components.mean(axis=1))
    return result


def _cliffs_delta(positive: pd.Series, negative: pd.Series) -> float:
    pos = positive.dropna().to_numpy(dtype=float)
    neg = negative.dropna().to_numpy(dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = np.concatenate([pos, neg])
    ranks = pd.Series(combined).rank(method="average").to_numpy()
    rank_sum_pos = ranks[: len(pos)].sum()
    u_statistic = rank_sum_pos - len(pos) * (len(pos) + 1) / 2
    return float((2 * u_statistic) / (len(pos) * len(neg)) - 1)


def _nearest_row(
    hive_frame: pd.DataFrame,
    timestamp: pd.Timestamp,
    *,
    tolerance_hours: float = 1.1,
) -> pd.Series | None:
    if hive_frame.empty:
        return None
    differences = (hive_frame[TIMESTAMP_COLUMN] - timestamp).abs()
    index = differences.idxmin()
    if differences.loc[index] > pd.Timedelta(hours=tolerance_hours):
        return None
    return hive_frame.loc[index]


def build_event_lead_samples(
    feature_frame: pd.DataFrame,
    events: pd.DataFrame,
    *,
    lead_hours: list[int],
    event_id_column: str = "harvest_event_id",
    event_start_column: str = "event_start",
) -> pd.DataFrame:
    """Create one independent sample per event and lead time."""
    _require_columns(
        events,
        {HIVE_COLUMN, event_id_column, event_start_column},
        frame_name="Harvest event table",
    )
    data = feature_frame.copy()
    data[TIMESTAMP_COLUMN] = pd.to_datetime(data[TIMESTAMP_COLUMN], errors="raise")
    event_frame = events.copy()
    event_frame[event_start_column] = pd.to_datetime(
        event_frame[event_start_column], errors="raise"
    )
    rows: list[dict[str, Any]] = []

    for event in event_frame.itertuples(index=False):
        values = event._asdict()
        hive_id = values[HIVE_COLUMN]
        event_start = pd.Timestamp(values[event_start_column])
        hive_frame = data.loc[data[HIVE_COLUMN].eq(hive_id)]
        for lead_hour in lead_hours:
            sample = _nearest_row(
                hive_frame,
                event_start - pd.Timedelta(hours=lead_hour),
            )
            if sample is None:
                continue
            row = {
                HIVE_COLUMN: hive_id,
                event_id_column: values[event_id_column],
                "event_start": event_start,
                "sample_timestamp": sample[TIMESTAMP_COLUMN],
                "lead_hours": lead_hour,
                "sample_type": "pre_harvest",
            }
            for feature in RELATIONSHIP_FEATURES:
                row[feature] = sample.get(feature, np.nan)
            for optional in ("split", "positive_rows", "event_duration_hours"):
                if optional in values:
                    row[optional] = values[optional]
            rows.append(row)
    return pd.DataFrame(rows)


def build_matched_control_samples(
    feature_frame: pd.DataFrame,
    events: pd.DataFrame,
    event_leads: pd.DataFrame,
    *,
    exclusion_hours: int,
    random_state: int,
    event_id_column: str = "harvest_event_id",
    event_start_column: str = "event_start",
) -> pd.DataFrame:
    """Match one non-event control to each event/lead sample."""
    data = feature_frame.copy()
    data[TIMESTAMP_COLUMN] = pd.to_datetime(data[TIMESTAMP_COLUMN], errors="raise")
    event_frame = events.copy()
    event_frame[event_start_column] = pd.to_datetime(
        event_frame[event_start_column], errors="raise"
    )
    event_times_by_hive = {
        hive_id: group[event_start_column].sort_values().tolist()
        for hive_id, group in event_frame.groupby(HIVE_COLUMN)
    }
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []

    for sample in event_leads.itertuples(index=False):
        values = sample._asdict()
        hive_id = values[HIVE_COLUMN]
        sample_timestamp = pd.Timestamp(values["sample_timestamp"])
        candidates = data.loc[data[HIVE_COLUMN].eq(hive_id)].copy()
        if "split" in candidates.columns and "split" in values:
            candidates = candidates.loc[
                candidates["split"].astype("string").eq(str(values["split"]))
            ]
        event_times = event_times_by_hive.get(hive_id, [])
        if event_times:
            keep = np.ones(len(candidates), dtype=bool)
            for event_time in event_times:
                keep &= (candidates[TIMESTAMP_COLUMN] - event_time).abs() > pd.Timedelta(
                    hours=exclusion_hours
                )
            candidates = candidates.loc[keep]

        same_month = candidates.loc[
            candidates[TIMESTAMP_COLUMN].dt.month.eq(sample_timestamp.month)
        ]
        if not same_month.empty:
            candidates = same_month
        candidates = candidates.dropna(subset=RELATIONSHIP_FEATURES, how="all")
        if candidates.empty:
            continue

        control = candidates.iloc[int(rng.integers(0, len(candidates)))]
        row = {
            HIVE_COLUMN: hive_id,
            event_id_column: values[event_id_column],
            "event_start": values["event_start"],
            "sample_timestamp": control[TIMESTAMP_COLUMN],
            "lead_hours": int(values["lead_hours"]),
            "sample_type": "matched_control",
        }
        for feature in RELATIONSHIP_FEATURES:
            row[feature] = control.get(feature, np.nan)
        for optional in ("split", "positive_rows", "event_duration_hours"):
            if optional in values:
                row[optional] = values[optional]
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_variable_relationships(
    event_leads: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    """Compare variables using event-level samples and matched controls."""
    rows: list[dict[str, Any]] = []
    for lead_hour in sorted(event_leads["lead_hours"].dropna().unique(), reverse=True):
        event_group = event_leads.loc[event_leads["lead_hours"].eq(lead_hour)]
        control_group = controls.loc[controls["lead_hours"].eq(lead_hour)]
        for feature in RELATIONSHIP_FEATURES:
            positive = event_group[feature].dropna()
            negative = control_group[feature].dropna()
            if len(positive) == 0 or len(negative) == 0:
                continue
            statistic, p_value = mannwhitneyu(
                positive,
                negative,
                alternative="two-sided",
            )
            delta = _cliffs_delta(positive, negative)
            rows.append(
                {
                    "lead_hours": int(lead_hour),
                    "feature": feature,
                    "event_count": len(positive),
                    "control_count": len(negative),
                    "event_median": positive.median(),
                    "control_median": negative.median(),
                    "median_difference": positive.median() - negative.median(),
                    "cliffs_delta": delta,
                    "absolute_effect_size": abs(delta),
                    "mann_whitney_u": statistic,
                    "p_value_descriptive": p_value,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["lead_hours", "absolute_effect_size"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _build_event_windows(
    feature_frame: pd.DataFrame,
    events: pd.DataFrame,
    *,
    pre_event_hours: int,
    post_event_hours: int,
    event_id_column: str,
    event_start_column: str,
) -> pd.DataFrame:
    windows: list[pd.DataFrame] = []
    event_frame = events.copy()
    event_frame[event_start_column] = pd.to_datetime(
        event_frame[event_start_column], errors="raise"
    )
    for event in event_frame.itertuples(index=False):
        values = event._asdict()
        hive_id = values[HIVE_COLUMN]
        event_start = pd.Timestamp(values[event_start_column])
        window = feature_frame.loc[
            feature_frame[HIVE_COLUMN].eq(hive_id)
            & feature_frame[TIMESTAMP_COLUMN].between(
                event_start - pd.Timedelta(hours=pre_event_hours),
                event_start + pd.Timedelta(hours=post_event_hours),
                inclusive="both",
            )
        ].copy()
        if window.empty:
            continue
        window["relative_hour"] = (
            ((window[TIMESTAMP_COLUMN] - event_start).dt.total_seconds() / 3600)
            .round()
            .astype("int32")
        )
        window[event_id_column] = values[event_id_column]
        window["event_start"] = event_start
        baseline = window.loc[window["relative_hour"].eq(-1), "weight_kg"]
        baseline_weight = float(baseline.iloc[0]) if not baseline.empty else np.nan
        window["weight_relative_to_pre_event_kg"] = window["weight_kg"] - baseline_weight
        windows.append(window)
    return pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_target_distribution(
    model_data: pd.DataFrame,
    *,
    target_column: str,
    output_path: Path,
) -> None:
    counts = model_data[target_column].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        ["No harvest within 72h", "Harvest within 72h"],
        [int(counts.get(0, 0)), int(counts.get(1, 0))],
    )
    axis.set_yscale("log")
    axis.set_ylabel("Rows (log scale)")
    axis.set_title("Severe imbalance in the 72-hour harvest target")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_path)


def _plot_events_per_hive(events: pd.DataFrame, *, output_path: Path) -> None:
    counts = events.groupby(HIVE_COLUMN).size().sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(counts.index.astype(str), counts.values)
    axis.set_title("Independent harvest events by hive")
    axis.set_xlabel("Hive")
    axis.set_ylabel("Consolidated event count")
    axis.tick_params(axis="x", rotation=75)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_path)


def _plot_events_by_month(
    events: pd.DataFrame,
    *,
    event_start_column: str,
    output_path: Path,
) -> None:
    times = pd.to_datetime(events[event_start_column], errors="raise")
    counts = times.dt.month.value_counts().reindex(range(1, 13), fill_value=0)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(counts.index, counts.values)
    axis.set_xticks(range(1, 13))
    axis.set_xlabel("Month")
    axis.set_ylabel("Independent event count")
    axis.set_title("Seasonal distribution of harvest markers")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_path)


def _plot_event_aligned_profile(
    windows: pd.DataFrame,
    *,
    column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    if windows.empty:
        return
    profile = (
        windows.groupby("relative_hour")[column]
        .agg(
            median="median",
            q25=lambda series: series.quantile(0.25),
            q75=lambda series: series.quantile(0.75),
        )
        .reset_index()
        .sort_values("relative_hour")
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    x_values = profile["relative_hour"].to_numpy()
    axis.plot(x_values, profile["median"].to_numpy(), label="Median")
    axis.fill_between(
        x_values,
        profile["q25"].to_numpy(),
        profile["q75"].to_numpy(),
        alpha=0.25,
        label="Interquartile range",
    )
    axis.axvline(0, linestyle="--", linewidth=1.4, label="Harvest marker")
    axis.set_xlabel("Hours relative to harvest marker")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.25)
    _save_figure(figure, output_path)


def _plot_effect_sizes(
    relationships: pd.DataFrame,
    *,
    lead_hour: int,
    output_path: Path,
) -> None:
    selected = relationships.loc[relationships["lead_hours"].eq(lead_hour)].copy()
    if selected.empty:
        return
    selected = selected.sort_values("cliffs_delta")
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.barh(selected["feature"], selected["cliffs_delta"])
    axis.axvline(0, linewidth=1)
    axis.set_xlabel("Cliff's delta: positive = higher before harvest")
    axis.set_title(f"Variable relationships at {lead_hour} hours before harvest")
    axis.grid(axis="x", alpha=0.25)
    _save_figure(figure, output_path)


def _plot_individual_events(
    windows: pd.DataFrame,
    *,
    event_id_column: str,
    output_directory: Path,
    maximum_plots: int,
) -> int:
    if windows.empty:
        return 0
    event_ids = windows[event_id_column].dropna().drop_duplicates().head(maximum_plots)
    output_directory.mkdir(parents=True, exist_ok=True)
    created = 0
    for event_id in event_ids:
        event_window = windows.loc[windows[event_id_column].eq(event_id)].sort_values(
            "relative_hour"
        )
        figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        definitions = [
            ("weight_kg", "Weight (kg)"),
            ("temperature_c", "Temperature (°C)"),
            ("humidity_pct", "Humidity (%)"),
            ("co2_ppm", "CO₂ (ppm)"),
        ]
        for axis, (column, label) in zip(axes, definitions, strict=True):
            axis.plot(event_window["relative_hour"], event_window[column])
            axis.axvline(0, linestyle="--", linewidth=1.1)
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
        axes[0].set_title(f"Sensor behaviour around {event_id}")
        axes[-1].set_xlabel("Hours relative to harvest marker")
        _save_figure(
            figure,
            output_directory / f"{_safe_filename(event_id)}.png",
        )
        created += 1
    return created


def run_harvest_research_eda(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Run research-grade, event-aware harvest EDA."""
    root = Path(backend_root).resolve()
    configuration_path = Path(config_path)
    if not configuration_path.is_absolute():
        configuration_path = root / configuration_path
    config = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))

    common_path = _resolve_path(root, config["dataset"]["clean_data_path"])
    model_path = _resolve_path(root, config["output"]["model_dataset_path"])
    event_path = _resolve_path(root, config["output"]["event_table_path"])
    eda_config = config["eda"]
    report_directory = _resolve_path(root, eda_config["report_directory"])
    report_directory.mkdir(parents=True, exist_ok=True)

    target_column = config["target"]["output_column"]
    event_id_column = config["event"]["event_id_column"]
    event_start_column = eda_config.get("event_table_start_column", "event_start")
    lead_hours = [int(value) for value in eda_config["lead_hours"]]

    common = pd.read_parquet(common_path)
    model_data = pd.read_parquet(model_path)
    events = pd.read_parquet(event_path)
    _require_columns(
        model_data,
        {HIVE_COLUMN, TIMESTAMP_COLUMN, target_column, "split"},
        frame_name="Harvest modelling dataset",
    )

    feature_frame = add_research_eda_features(common)
    split_columns = model_data[[HIVE_COLUMN, TIMESTAMP_COLUMN, "split"]].copy()
    split_columns[TIMESTAMP_COLUMN] = pd.to_datetime(
        split_columns[TIMESTAMP_COLUMN], errors="raise"
    )
    feature_frame = feature_frame.merge(
        split_columns,
        on=[HIVE_COLUMN, TIMESTAMP_COLUMN],
        how="left",
        validate="one_to_one",
    )

    event_leads = build_event_lead_samples(
        feature_frame,
        events,
        lead_hours=lead_hours,
        event_id_column=event_id_column,
        event_start_column=event_start_column,
    )
    controls = build_matched_control_samples(
        feature_frame,
        events,
        event_leads,
        exclusion_hours=int(eda_config["exclusion_hours_for_controls"]),
        random_state=int(eda_config["random_state"]),
        event_id_column=event_id_column,
        event_start_column=event_start_column,
    )
    relationships = calculate_variable_relationships(event_leads, controls)
    windows = _build_event_windows(
        feature_frame,
        events,
        pre_event_hours=int(eda_config["pre_event_hours"]),
        post_event_hours=int(eda_config["post_event_hours"]),
        event_id_column=event_id_column,
        event_start_column=event_start_column,
    )

    event_leads.to_csv(report_directory / "event_lead_samples.csv", index=False)
    controls.to_csv(report_directory / "matched_control_samples.csv", index=False)
    relationships.to_csv(report_directory / "variable_relationships.csv", index=False)
    if not windows.empty:
        windows.to_parquet(report_directory / "event_windows.parquet", index=False)

    events_by_hive = (
        events.groupby(HIVE_COLUMN)
        .size()
        .rename("event_count")
        .reset_index()
        .sort_values("event_count", ascending=False)
    )
    events_by_hive.to_csv(report_directory / "events_by_hive.csv", index=False)
    event_times = pd.to_datetime(events[event_start_column], errors="raise")
    events_by_month = (
        event_times.dt.month.value_counts()
        .reindex(range(1, 13), fill_value=0)
        .rename_axis("month")
        .rename("event_count")
        .reset_index()
    )
    events_by_month.to_csv(report_directory / "events_by_month.csv", index=False)

    figure_directory = report_directory / "figures"
    _plot_target_distribution(
        model_data,
        target_column=target_column,
        output_path=figure_directory / "target_distribution_log_scale.png",
    )
    _plot_events_per_hive(
        events,
        output_path=figure_directory / "events_per_hive.png",
    )
    _plot_events_by_month(
        events,
        event_start_column=event_start_column,
        output_path=figure_directory / "events_by_month.png",
    )
    _plot_event_aligned_profile(
        windows,
        column="weight_relative_to_pre_event_kg",
        ylabel="Weight relative to one hour before event (kg)",
        title="Event-aligned hive-weight profile around harvest markers",
        output_path=figure_directory / "event_aligned_weight_profile.png",
    )
    for column, ylabel, title, filename in [
        (
            "temperature_c",
            "Internal temperature (°C)",
            "Temperature around harvest markers",
            "event_aligned_temperature_profile.png",
        ),
        (
            "humidity_pct",
            "Humidity (%)",
            "Humidity around harvest markers",
            "event_aligned_humidity_profile.png",
        ),
        ("co2_ppm", "CO₂ (ppm)", "CO₂ around harvest markers", "event_aligned_co2_profile.png"),
    ]:
        _plot_event_aligned_profile(
            windows,
            column=column,
            ylabel=ylabel,
            title=title,
            output_path=figure_directory / filename,
        )
    for lead_hour in lead_hours:
        _plot_effect_sizes(
            relationships,
            lead_hour=lead_hour,
            output_path=figure_directory / f"effect_sizes_{lead_hour}h_before.png",
        )
    individual_plot_count = _plot_individual_events(
        windows,
        event_id_column=event_id_column,
        output_directory=figure_directory / "individual_events",
        maximum_plots=int(eda_config["maximum_individual_event_plots"]),
    )

    top_relationships: dict[str, list[dict[str, Any]]] = {}
    for lead_hour in lead_hours:
        selected = relationships.loc[relationships["lead_hours"].eq(lead_hour)].nlargest(
            5, "absolute_effect_size"
        )
        top_relationships[str(lead_hour)] = selected[
            [
                "feature",
                "event_median",
                "control_median",
                "cliffs_delta",
                "p_value_descriptive",
            ]
        ].to_dict(orient="records")

    summary = {
        "modelling_rows": len(model_data),
        "target_positive_rows": int(model_data[target_column].sum()),
        "target_positive_rate": float(model_data[target_column].mean()),
        "independent_events": len(events),
        "positive_hives": int(events[HIVE_COLUMN].nunique()),
        "event_lead_samples": len(event_leads),
        "matched_control_samples": len(controls),
        "lead_hours_analysed": lead_hours,
        "individual_event_plots_created": individual_plot_count,
        "events_by_split": (
            events.groupby("split", observed=True).size().to_dict()
            if "split" in events.columns
            else {}
        ),
        "top_variable_relationships_by_lead": top_relationships,
        "research_interpretation_rules": {
            "weight": (
                "Weight accumulation, proximity to the recent maximum and "
                "low short-term variability are examined as direct readiness signals."
            ),
            "temperature": (
                "Temperature is interpreted as colony/environmental stability context, "
                "not as a direct honey-volume measurement."
            ),
            "humidity": (
                "Humidity is interpreted as moisture context. In this dataset it is an "
                "estimated value and must not be presented as directly measured."
            ),
            "co2": (
                "CO₂ is interpreted as ventilation and colony-activity context, not as "
                "a direct honey-storage measurement."
            ),
        },
        "limitations": [
            "Harvest markers are generated labels rather than beekeeper-confirmed events.",
            "Only 44 independent events are available, with very few in validation and test.",
            "Hourly positive rows are correlated; comparisons therefore use one event sample per event and lead time.",
            "P-values are descriptive; effect sizes, event plots and biological plausibility are more important.",
        ],
    }
    (report_directory / "harvest_eda_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2),
        encoding="utf-8",
    )
    return summary
