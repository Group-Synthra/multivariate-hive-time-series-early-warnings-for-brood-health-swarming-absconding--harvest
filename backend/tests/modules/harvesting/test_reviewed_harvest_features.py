import pandas as pd
from pandas.testing import assert_frame_equal

from multivari.modules.harvesting.reviewed_features import (
    BANNED_MODEL_COLUMNS,
    build_reviewed_feature_dataset,
)


def _source_frame(
    *,
    periods: int = 500,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-01-01",
        periods=periods,
        freq="h",
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "hive_id": ["h1"] * periods,
            "split": ["train"] * periods,
            "harvest_within_next_72h_reviewed": ([0] * (periods - 72) + [1] * 72),
            "weight_kg": [50.0 + index * 0.01 for index in range(periods)],
            "temperature_c": [30.0 + (index % 24) * 0.01 for index in range(periods)],
            "humidity_pct": [60.0 + (index % 12) * 0.02 for index in range(periods)],
            "co2_ppm": [500.0 + (index % 20) for index in range(periods)],
            "honey_harvested_1": [0] * periods,
            "harvest_reviewed_event_start_1": [0] * periods,
        }
    )


def _build(
    history: pd.DataFrame,
    modelling: pd.DataFrame | None = None,
) -> pd.DataFrame:
    output, _, _ = build_reviewed_feature_dataset(
        history,
        modelling,
        target_column="harvest_within_next_72h_reviewed",
        minimum_history_hours=168,
        weight_windows_hours=[6, 24, 72, 168],
        environmental_windows_hours=[24, 72],
        weight_delta_hours=[1, 6, 24, 72],
        environmental_delta_hours=[1, 6, 24],
        weight_trend_hours=[6, 24, 72],
        environmental_trend_hours=[24],
        co2_flatline_std_threshold=1.0,
    )
    return output


def test_feature_output_excludes_leakage_columns() -> None:
    output = _build(_source_frame())

    assert not set(output.columns).intersection(BANNED_MODEL_COLUMNS)


def test_future_changes_do_not_change_past_features() -> None:
    source = _source_frame()
    changed = source.copy()
    cutoff = source.loc[300, "timestamp"]

    changed.loc[
        changed["timestamp"].gt(cutoff),
        "weight_kg",
    ] = 9999.0
    changed.loc[
        changed["timestamp"].gt(cutoff),
        "co2_ppm",
    ] = 9999.0

    original_features = _build(source)
    changed_features = _build(changed)

    feature_columns = [
        column
        for column in original_features.columns
        if column
        not in {
            "timestamp",
            "hive_id",
            "split",
            "harvest_within_next_72h_reviewed",
        }
    ]

    original_past = original_features.loc[
        original_features["timestamp"].le(cutoff),
        feature_columns,
    ].reset_index(drop=True)
    changed_past = changed_features.loc[
        changed_features["timestamp"].le(cutoff),
        feature_columns,
    ].reset_index(drop=True)

    assert_frame_equal(
        original_past,
        changed_past,
        check_dtype=False,
    )


def test_features_do_not_cross_non_hourly_gap() -> None:
    first = _source_frame(periods=220)
    second = _source_frame(periods=220)
    second["timestamp"] = pd.date_range(
        "2024-02-01",
        periods=220,
        freq="h",
    )
    second["weight_kg"] = 500.0

    source = pd.concat(
        [first, second],
        ignore_index=True,
    )
    output = _build(source)

    second_start = second["timestamp"].min()
    first_kept_second_segment = output.loc[
        output["timestamp"].ge(second_start),
        "timestamp",
    ].min()

    assert first_kept_second_segment == (second_start + pd.Timedelta(hours=167))


def test_all_features_are_finite_after_filtering() -> None:
    output = _build(_source_frame())

    feature_columns = output.columns.difference(
        [
            "timestamp",
            "hive_id",
            "split",
            "harvest_within_next_72h_reviewed",
        ]
    )

    assert output[feature_columns].notna().all().all()


def test_removed_modelling_rows_do_not_reset_sensor_history() -> None:
    history = _source_frame(periods=500)
    modelling = history.drop(index=range(220, 245)).reset_index(drop=True)

    output = _build(history, modelling)

    first_row_after_removed_period = history.loc[
        245,
        "timestamp",
    ]
    assert first_row_after_removed_period in set(output["timestamp"])
