# from __future__ import annotations

# from dataclasses import dataclass

# import pandas as pd


# @dataclass(frozen=True)
# class HarvestFold:
#     fold: int
#     train_start: pd.Timestamp
#     train_end: pd.Timestamp
#     validation_start: pd.Timestamp
#     validation_end: pd.Timestamp
#     training_events: int
#     validation_events: int


# def create_event_aware_folds(
#     events: pd.DataFrame,
#     *,
#     split_column: str = "split",
#     event_start_column: str = "event_start",
#     minimum_training_events: int = 12,
#     validation_events_per_fold: int = 6,
#     purge_hours: int = 72,
# ) -> list[HarvestFold]:
#     """Create expanding chronological folds from training events."""

#     required = {
#         split_column,
#         event_start_column,
#     }

#     missing = sorted(required.difference(events.columns))

#     if missing:
#         raise ValueError(
#             f"Missing columns for event-aware splitting: {missing}"
#         )

#     training_events = events.loc[
#         events[split_column].eq("train")
#     ].copy()

#     training_events[event_start_column] = pd.to_datetime(
#         training_events[event_start_column],
#         errors="raise",
#     )

#     training_events = training_events.sort_values(
#         event_start_column
#     ).reset_index(drop=True)

#     event_count = len(training_events)

#     if event_count < (
#         minimum_training_events
#         + validation_events_per_fold
#     ):
#         raise ValueError(
#             "Not enough training events to create rolling folds."
#         )

#     purge = pd.Timedelta(hours=purge_hours)
#     folds: list[HarvestFold] = []

#     validation_begin = minimum_training_events
#     fold_number = 1

#     while validation_begin < event_count:
#         validation_end_index = min(
#             validation_begin + validation_events_per_fold,
#             event_count,
#         )

#         validation_group = training_events.iloc[
#             validation_begin:validation_end_index
#         ]

#         if validation_group.empty:
#             break

#         first_validation_event = validation_group[
#             event_start_column
#         ].min()

#         last_validation_event = validation_group[
#             event_start_column
#         ].max()

#         # Include the hours in which the model should predict the
#         # first event in the validation group.
#         validation_start = (
#             first_validation_event - purge
#         )

#         # Remove a further 72 hours from training so a training
#         # target cannot look forward into the validation period.
#         train_end = validation_start - purge

#         eligible_training = training_events.loc[
#             training_events[event_start_column] <= train_end
#         ]

#         if len(eligible_training) < minimum_training_events:
#             validation_begin += validation_events_per_fold
#             continue

#         folds.append(
#             HarvestFold(
#                 fold=fold_number,
#                 train_start=training_events[
#                     event_start_column
#                 ].min(),
#                 train_end=train_end,
#                 validation_start=validation_start,
#                 validation_end=last_validation_event,
#                 training_events=len(eligible_training),
#                 validation_events=len(validation_group),
#             )
#         )

#         fold_number += 1
#         validation_begin += validation_events_per_fold

#     return folds



from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class HarvestFold:
    """Time boundaries and event counts for one validation fold."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    training_events: int
    validation_events: int


def create_event_aware_folds(
    events: pd.DataFrame,
    *,
    split_column: str = "split",
    event_start_column: str = "event_start",
    minimum_training_events: int = 12,
    validation_events_per_fold: int = 6,
    prediction_horizon_hours: int = 72,
    purge_hours: int = 72,
) -> list[HarvestFold]:
    """
    Create expanding chronological folds from official training events.

    Validation periods are based on independent harvest events. A purge
    interval prevents a training target from looking forward into the
    validation prediction window.
    """
    required_columns = {
        split_column,
        event_start_column,
    }

    missing_columns = sorted(
        required_columns.difference(events.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns required for event-aware splitting: "
            f"{missing_columns}"
        )

    if minimum_training_events <= 0:
        raise ValueError(
            "minimum_training_events must be greater than zero"
        )

    if validation_events_per_fold <= 0:
        raise ValueError(
            "validation_events_per_fold must be greater than zero"
        )

    if prediction_horizon_hours <= 0:
        raise ValueError(
            "prediction_horizon_hours must be greater than zero"
        )

    if purge_hours < 0:
        raise ValueError(
            "purge_hours cannot be negative"
        )

    training_events = events.loc[
        events[split_column].astype("string").eq("train")
    ].copy()

    if "is_boundary_gap" in training_events.columns:
        training_events = training_events.loc[
            ~training_events["is_boundary_gap"]
            .fillna(False)
            .astype(bool)
        ].copy()

    training_events[event_start_column] = pd.to_datetime(
        training_events[event_start_column],
        errors="raise",
    )

    training_events = (
        training_events.sort_values(event_start_column)
        .reset_index(drop=True)
    )

    required_event_count = (
        minimum_training_events
        + validation_events_per_fold
    )

    if len(training_events) < required_event_count:
        raise ValueError(
            "Not enough official training events to construct folds. "
            f"Required at least {required_event_count}, "
            f"but found {len(training_events)}."
        )

    prediction_horizon = pd.Timedelta(
        hours=prediction_horizon_hours
    )
    purge_interval = pd.Timedelta(hours=purge_hours)

    folds: list[HarvestFold] = []

    validation_begin_index = minimum_training_events
    fold_number = 1

    while validation_begin_index < len(training_events):
        validation_end_index = min(
            validation_begin_index
            + validation_events_per_fold,
            len(training_events),
        )

        validation_group = training_events.iloc[
            validation_begin_index:validation_end_index
        ]

        if validation_group.empty:
            break

        first_validation_event = validation_group[
            event_start_column
        ].min()

        last_validation_event = validation_group[
            event_start_column
        ].max()

        # Include the complete future-prediction period before the
        # first validation event.
        validation_start = (
            first_validation_event
            - prediction_horizon
        )

        # Training ends before the purge interval. This prevents
        # a training target from looking forward into validation.
        train_end = (
            validation_start
            - purge_interval
        )

        eligible_training_events = training_events.loc[
            training_events[event_start_column] <= train_end
        ]

        if (
            len(eligible_training_events)
            >= minimum_training_events
        ):
            folds.append(
                HarvestFold(
                    fold=fold_number,
                    train_start=eligible_training_events[
                        event_start_column
                    ].min(),
                    train_end=train_end,
                    validation_start=validation_start,
                    validation_end=last_validation_event,
                    training_events=len(
                        eligible_training_events
                    ),
                    validation_events=len(
                        validation_group
                    ),
                )
            )

            fold_number += 1

        validation_begin_index += (
            validation_events_per_fold
        )

    if not folds:
        raise ValueError(
            "No valid folds could be created after applying the "
            "prediction horizon and purge interval."
        )

    return folds


def folds_to_frame(
    folds: list[HarvestFold],
) -> pd.DataFrame:
    """Convert fold objects into a DataFrame for reporting."""
    if not folds:
        return pd.DataFrame(
            columns=[
                "fold",
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "training_events",
                "validation_events",
            ]
        )

    return pd.DataFrame(
        [asdict(fold) for fold in folds]
    )