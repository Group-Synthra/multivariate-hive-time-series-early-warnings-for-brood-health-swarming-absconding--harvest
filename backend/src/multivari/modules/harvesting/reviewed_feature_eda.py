from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

HIVE_COLUMN = "hive_id"
TIMESTAMP_COLUMN = "timestamp"
SPLIT_COLUMN = "split"


def _resolve_path(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    frame_name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: {missing}"
        )


def _event_distance_hours(
    timestamps: pd.Series,
    event_times: np.ndarray,
) -> np.ndarray:
    values = timestamps.to_numpy(dtype="datetime64[ns]")
    if len(event_times) == 0:
        return np.full(len(values), np.inf)

    positions = np.searchsorted(event_times, values)
    distances = np.full(len(values), np.inf)

    previous_valid = positions > 0
    if previous_valid.any():
        previous = event_times[positions[previous_valid] - 1]
        distances[previous_valid] = np.minimum(
            distances[previous_valid],
            np.abs(
                (
                    values[previous_valid] - previous
                )
                / np.timedelta64(1, "h")
            ),
        )

    next_valid = positions < len(event_times)
    if next_valid.any():
        following = event_times[positions[next_valid]]
        distances[next_valid] = np.minimum(
            distances[next_valid],
            np.abs(
                (
                    following - values[next_valid]
                )
                / np.timedelta64(1, "h")
            ),
        )

    return distances


def create_event_and_control_samples(
    features: pd.DataFrame,
    events: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    lead_hours: list[int],
    control_exclusion_hours: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(
        features,
        {
            HIVE_COLUMN,
            TIMESTAMP_COLUMN,
            SPLIT_COLUMN,
            target_column,
            *feature_columns,
        },
        frame_name="Feature dataset",
    )
    _require_columns(
        events,
        {
            HIVE_COLUMN,
            "harvest_event_id",
            "event_start",
            SPLIT_COLUMN,
        },
        frame_name="Reviewed events",
    )

    feature_frame = features.copy()
    feature_frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        feature_frame[TIMESTAMP_COLUMN],
        errors="raise",
    )
    event_frame = events.copy()
    event_frame["event_start"] = pd.to_datetime(
        event_frame["event_start"],
        errors="raise",
    )

    indexed = feature_frame.set_index(
        [HIVE_COLUMN, TIMESTAMP_COLUMN]
    )
    rng = np.random.default_rng(random_state)

    event_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    used_control_keys: set[tuple[str, pd.Timestamp]] = set()

    events_by_hive = {
        hive_id: group["event_start"]
        .sort_values()
        .to_numpy(dtype="datetime64[ns]")
        for hive_id, group in event_frame.groupby(
            HIVE_COLUMN,
            sort=False,
        )
    }

    for event in event_frame.itertuples(index=False):
        hive_id = getattr(event, HIVE_COLUMN)
        event_id = event.harvest_event_id
        event_start = event.event_start
        event_split = event.split

        hive_candidates = feature_frame.loc[
            feature_frame[HIVE_COLUMN].eq(hive_id)
            & feature_frame[SPLIT_COLUMN].eq(event_split)
            & feature_frame[target_column].eq(0)
        ].copy()

        event_times = events_by_hive[hive_id]
        hive_candidates["_event_distance_hours"] = (
            _event_distance_hours(
                hive_candidates[TIMESTAMP_COLUMN],
                event_times,
            )
        )
        hive_candidates = hive_candidates.loc[
            hive_candidates["_event_distance_hours"].gt(
                control_exclusion_hours
            )
        ]

        for lead in lead_hours:
            sample_time = event_start - pd.Timedelta(hours=lead)
            key = (hive_id, sample_time)

            event_available = key in indexed.index
            coverage = {
                "harvest_event_id": event_id,
                HIVE_COLUMN: hive_id,
                SPLIT_COLUMN: event_split,
                "event_start": event_start,
                "lead_hours": lead,
                "sample_time": sample_time,
                "event_sample_available": bool(event_available),
                "control_sample_available": False,
            }

            if not event_available:
                coverage_rows.append(coverage)
                continue

            event_sample = indexed.loc[key]
            if isinstance(event_sample, pd.DataFrame):
                raise TypeError(
                    "Duplicate feature rows found for an event lead."
                )

            event_record = {
                "harvest_event_id": event_id,
                HIVE_COLUMN: hive_id,
                SPLIT_COLUMN: event_split,
                "event_start": event_start,
                "lead_hours": lead,
                "sample_time": sample_time,
            }
            for feature in feature_columns:
                event_record[feature] = event_sample[feature]
            event_rows.append(event_record)

            same_hour = hive_candidates.loc[
                hive_candidates[TIMESTAMP_COLUMN].dt.hour.eq(
                    sample_time.hour
                )
            ]
            unused = same_hour.loc[
                ~same_hour.apply(
                    lambda row: (
                        row[HIVE_COLUMN],
                        row[TIMESTAMP_COLUMN],
                    )
                    in used_control_keys,
                    axis=1,
                )
            ]
            pool = unused if not unused.empty else same_hour

            if pool.empty:
                coverage_rows.append(coverage)
                continue

            chosen_position = int(
                rng.integers(0, len(pool))
            )
            chosen = pool.iloc[chosen_position]
            control_key = (
                chosen[HIVE_COLUMN],
                chosen[TIMESTAMP_COLUMN],
            )
            used_control_keys.add(control_key)

            control_record = {
                "harvest_event_id": event_id,
                HIVE_COLUMN: hive_id,
                SPLIT_COLUMN: event_split,
                "event_start": event_start,
                "lead_hours": lead,
                "sample_time": chosen[TIMESTAMP_COLUMN],
                "matched_event_sample_time": sample_time,
                "distance_from_nearest_event_hours": chosen[
                    "_event_distance_hours"
                ],
            }
            for feature in feature_columns:
                control_record[feature] = chosen[feature]
            control_rows.append(control_record)

            coverage["control_sample_available"] = True
            coverage_rows.append(coverage)

    return (
        pd.DataFrame(event_rows),
        pd.DataFrame(control_rows),
        pd.DataFrame(coverage_rows),
    )


def compare_event_and_control_features(
    event_samples: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    common_leads = sorted(
        set(event_samples["lead_hours"])
        .intersection(controls["lead_hours"])
    )

    for lead in common_leads:
        event_lead = event_samples.loc[
            event_samples["lead_hours"].eq(lead)
        ]
        control_lead = controls.loc[
            controls["lead_hours"].eq(lead)
        ]

        for feature in feature_columns:
            event_values = pd.to_numeric(
                event_lead[feature],
                errors="coerce",
            ).dropna()
            control_values = pd.to_numeric(
                control_lead[feature],
                errors="coerce",
            ).dropna()

            event_mean = float(event_values.mean())
            control_mean = float(control_values.mean())
            mean_difference = event_mean - control_mean
            pooled_standard_deviation = float(
                np.sqrt(
                    (
                        event_values.var(ddof=1)
                        + control_values.var(ddof=1)
                    )
                    / 2
                )
            )
            if (
                np.isfinite(pooled_standard_deviation)
                and pooled_standard_deviation > 0
            ):
                standardized_mean_difference = (
                    mean_difference
                    / pooled_standard_deviation
                )
            else:
                standardized_mean_difference = np.nan

            records.append(
                {
                    "lead_hours": lead,
                    "feature": feature,
                    "event_n": len(event_values),
                    "control_n": len(control_values),
                    "event_mean": event_mean,
                    "control_mean": control_mean,
                    "event_median": float(
                        event_values.median()
                    ),
                    "control_median": float(
                        control_values.median()
                    ),
                    "mean_difference": mean_difference,
                    "standardized_mean_difference": (
                        standardized_mean_difference
                    ),
                    "absolute_standardized_mean_difference": (
                        abs(standardized_mean_difference)
                        if np.isfinite(
                            standardized_mean_difference
                        )
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(records)


def _save_top_feature_plots(
    comparison: pd.DataFrame,
    *,
    output_directory: Path,
    top_features_to_plot: int,
) -> list[str]:
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure_paths: list[str] = []

    for lead in sorted(comparison["lead_hours"].unique()):
        subset = (
            comparison.loc[
                comparison["lead_hours"].eq(lead)
            ]
            .dropna(
                subset=[
                    "absolute_standardized_mean_difference"
                ]
            )
            .nlargest(
                top_features_to_plot,
                "absolute_standardized_mean_difference",
            )
            .sort_values(
                "absolute_standardized_mean_difference"
            )
        )
        if subset.empty:
            continue

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.barh(
            subset["feature"],
            subset[
                "absolute_standardized_mean_difference"
            ],
        )
        axis.set_title(
            f"Top reviewed-event feature differences at {lead} h lead"
        )
        axis.set_xlabel(
            "Absolute standardized mean difference"
        )
        axis.set_ylabel("Feature")
        figure.tight_layout()

        path = (
            figure_directory
            / f"top_features_lead_{lead}h.png"
        )
        figure.savefig(path, dpi=160)
        plt.close(figure)
        figure_paths.append(str(path))

    return figure_paths


def run_reviewed_feature_eda_from_config(
    *,
    backend_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(backend_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    config = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    settings = config["reviewed_feature_eda"]
    target_column = config["reviewed_target"]["output_column"]

    event_path = _resolve_path(
        root,
        settings["event_table_path"],
    )
    feature_path = _resolve_path(
        root,
        settings["feature_dataset_path"],
    )
    manifest_path = _resolve_path(
        root,
        settings["feature_manifest_path"],
    )
    output_directory = _resolve_path(
        root,
        settings["report_directory"],
    )

    events = pd.read_parquet(event_path)
    features = pd.read_parquet(feature_path)
    manifest = pd.read_csv(manifest_path)
    feature_columns = manifest["feature_name"].tolist()

    event_samples, controls, coverage = (
        create_event_and_control_samples(
            features,
            events,
            feature_columns=feature_columns,
            target_column=target_column,
            lead_hours=[
                int(value)
                for value in settings["lead_hours"]
            ],
            control_exclusion_hours=int(
                settings["control_exclusion_hours"]
            ),
            random_state=int(settings["random_state"]),
        )
    )
    comparison = compare_event_and_control_features(
        event_samples,
        controls,
        feature_columns=feature_columns,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    event_samples.to_csv(
        output_directory / "event_feature_samples.csv",
        index=False,
    )
    controls.to_csv(
        output_directory / "matched_control_samples.csv",
        index=False,
    )
    coverage.to_csv(
        output_directory / "sample_coverage.csv",
        index=False,
    )
    comparison.to_csv(
        output_directory / "lead_feature_comparison.csv",
        index=False,
    )

    top_features = (
        comparison.dropna(
            subset=["absolute_standardized_mean_difference"]
        )
        .sort_values(
            [
                "lead_hours",
                "absolute_standardized_mean_difference",
            ],
            ascending=[True, False],
        )
        .groupby("lead_hours", observed=True)
        .head(
            int(settings["top_features_to_plot"])
        )
    )
    top_features.to_csv(
        output_directory / "top_features_by_lead.csv",
        index=False,
    )

    figure_paths = _save_top_feature_plots(
        comparison,
        output_directory=output_directory,
        top_features_to_plot=int(
            settings["top_features_to_plot"]
        ),
    )

    expected_samples = len(events) * len(
        settings["lead_hours"]
    )
    audit = {
        "reviewed_event_count": len(events),
        "expected_event_lead_samples": expected_samples,
        "available_event_lead_samples": len(event_samples),
        "available_matched_controls": len(controls),
        "missing_event_lead_samples": int(
            expected_samples - len(event_samples)
        ),
        "missing_controls": int(
            expected_samples - len(controls)
        ),
        "feature_count": len(feature_columns),
        "lead_hours": [
            int(value)
            for value in settings["lead_hours"]
        ],
        "control_exclusion_hours": int(
            settings["control_exclusion_hours"]
        ),
        "figure_count": len(figure_paths),
        "figure_paths": figure_paths,
        "warning": (
            "Only 12 probable pseudo-events are available. "
            "Standardized differences are exploratory and must not "
            "be treated as confirmatory statistical evidence."
        ),
    }
    (
        output_directory / "reviewed_feature_eda_audit.json"
    ).write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    return {
        **audit,
        "report_directory": str(output_directory),
    }
