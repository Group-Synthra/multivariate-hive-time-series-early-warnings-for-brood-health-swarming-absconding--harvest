from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multivari.common.schema import (
    HIVE_COLUMN,
    SENSOR_COLUMNS,
    TARGET_COLUMNS,
    TIMESTAMP_COLUMN,
)

SENSOR_API_NAMES = {
    "temperature_c": "temperature",
    "humidity_pct": "humidity",
    "co2_ppm": "co2",
    "weight_kg": "weight",
}

TARGET_DISPLAY_NAMES = {
    "brood_health_healthy_1": "Brood healthy",
    "swarming_happened_1": "Swarming",
    "absconding_happened_1": "Absconding",
    "honey_harvested_1": "Harvesting",
}

ALLOWED_EDA_IMAGES = {
    "target_balance.png",
    "sensor_distributions.png",
    "rows_per_hive.png",
    "correlation_heatmap.png",
    "hourly_sensor_patterns.png",
    "monthly_sensor_trends.png",
    "outlier_percentages.png",
    "target_positive_rates_log.png",
    "monthly_event_timeline.png",
    "target_cooccurrence.png",
}


@dataclass
class EDAService:
    backend_root: Path
    processed_path: Path = field(init=False)
    report_directory: Path = field(init=False)
    _cache_signature: tuple[int, int] | None = field(default=None, init=False)
    _cache_payload: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.backend_root = Path(self.backend_root).resolve()
        self.processed_path = self.backend_root / "data" / "processed" / "common_clean.parquet"
        self.report_directory = self.backend_root / "artifacts" / "reports" / "common_eda"

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "multivari-backend",
            "processed_dataset_ready": self.processed_path.exists(),
            "processed_dataset": str(self.processed_path),
        }

    def get_common_eda(self) -> dict[str, Any]:
        if not self.processed_path.exists():
            raise FileNotFoundError(
                "Processed dataset was not found. Run: "
                "python scripts/run_common_pipeline.py --input "
                "data/raw/Common_Beehive_Complete_Training_Dataset_311044.xlsx"
            )

        stat = self.processed_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if self._cache_signature == signature and self._cache_payload is not None:
            return self._cache_payload

        columns = [
            TIMESTAMP_COLUMN,
            HIVE_COLUMN,
            *SENSOR_COLUMNS,
            *TARGET_COLUMNS,
        ]
        df = pd.read_parquet(self.processed_path, columns=columns)
        df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")

        payload = {
            "summary": self._build_summary(df),
            "sensor_statistics": self._build_sensor_statistics(df),
            "outlier_analysis": self._build_outlier_analysis(df),
            "hive_stats": self._build_hive_stats(df),
            "hourly_patterns": self._build_hourly_patterns(df),
            "weekday_patterns": self._build_weekday_patterns(df),
            "monthly_patterns": self._build_monthly_patterns(df),
            "sensor_histograms": self._build_sensor_histograms(df),
            "target_balance": self._build_target_balance(df),
            "target_by_hive": self._build_target_by_hive(df),
            "monthly_target_counts": self._build_monthly_target_counts(df),
            "target_cooccurrence": self._build_target_cooccurrence(df),
            "target_sensor_effects": self._build_target_sensor_effects(df),
            "relationship_sample": self._build_relationship_sample(df),
            "correlation": self._build_correlation(df),
            "generated_images": self._generated_images(),
            "data_quality": self._build_data_quality(df),
        }

        self._cache_signature = signature
        self._cache_payload = payload
        return payload

    def image_path(self, filename: str) -> Path:
        if filename not in ALLOWED_EDA_IMAGES:
            raise FileNotFoundError(f"Unknown EDA image: {filename}")

        path = (self.report_directory / filename).resolve()
        if path.parent != self.report_directory.resolve() or not path.is_file():
            raise FileNotFoundError(f"EDA image was not found: {filename}")
        return path

    @staticmethod
    def _build_summary(df: pd.DataFrame) -> dict[str, Any]:
        valid_timestamps = df[TIMESTAMP_COLUMN].dropna()
        start = valid_timestamps.min() if not valid_timestamps.empty else None
        end = valid_timestamps.max() if not valid_timestamps.empty else None
        duration_days = None
        if start is not None and end is not None:
            duration_days = round((end - start).total_seconds() / 86400, 2)

        return {
            "total_records": len(df),
            "total_hives": int(df[HIVE_COLUMN].nunique(dropna=True)),
            "analysis_start": start.isoformat() if start is not None else None,
            "analysis_end": end.isoformat() if end is not None else None,
            "duration_days": duration_days,
            "sampling_frequency": EDAService._infer_sampling_frequency(df),
        }

    @staticmethod
    def _infer_sampling_seconds(df: pd.DataFrame) -> float | None:
        ordered = (
            df[[HIVE_COLUMN, TIMESTAMP_COLUMN]]
            .dropna()
            .sort_values([HIVE_COLUMN, TIMESTAMP_COLUMN])
        )
        differences = ordered.groupby(HIVE_COLUMN)[TIMESTAMP_COLUMN].diff().dropna()
        if differences.empty:
            return None
        value = float(differences.dt.total_seconds().median())
        return value if value > 0 else None

    @staticmethod
    def _infer_sampling_frequency(df: pd.DataFrame) -> str:
        median_seconds = EDAService._infer_sampling_seconds(df)
        if median_seconds is None:
            return "unknown"
        if median_seconds % 3600 == 0:
            hours = median_seconds / 3600
            return f"{hours:g} hour" + ("s" if hours != 1 else "")
        if median_seconds % 60 == 0:
            minutes = median_seconds / 60
            return f"{minutes:g} minutes"
        return f"{median_seconds:g} seconds"

    @staticmethod
    def _build_sensor_statistics(df: pd.DataFrame) -> dict[str, Any]:
        summary = df[list(SENSOR_COLUMNS)].describe(percentiles=[0.25, 0.5, 0.75]).T
        result: dict[str, Any] = {}

        for source_name, api_name in SENSOR_API_NAMES.items():
            row = summary.loc[source_name]
            result[api_name] = {
                "count": int(row["count"]),
                "mean": EDAService._round(row["mean"]),
                "std": EDAService._round(row["std"]),
                "min": EDAService._round(row["min"]),
                "q1": EDAService._round(row["25%"]),
                "median": EDAService._round(row["50%"]),
                "q3": EDAService._round(row["75%"]),
                "max": EDAService._round(row["max"]),
            }

        return result

    @staticmethod
    def _build_target_balance(df: pd.DataFrame) -> list[dict[str, Any]]:
        total = len(df)
        result = []
        for target in TARGET_COLUMNS:
            positive = int((df[target] == 1).sum())
            negative = int((df[target] == 0).sum())
            result.append(
                {
                    "target": target,
                    "display_name": TARGET_DISPLAY_NAMES.get(target, target),
                    "positive": positive,
                    "negative": negative,
                    "positive_percentage": round((positive / total) * 100, 8) if total else 0.0,
                    "positive_per_10000": round((positive / total) * 10000, 4) if total else 0.0,
                }
            )
        return result

    @staticmethod
    def _build_hive_stats(df: pd.DataFrame) -> list[dict[str, Any]]:
        sampling_seconds = EDAService._infer_sampling_seconds(df)
        grouped = (
            df.groupby(HIVE_COLUMN, dropna=False)
            .agg(
                records=(TIMESTAMP_COLUMN, "size"),
                start=(TIMESTAMP_COLUMN, "min"),
                end=(TIMESTAMP_COLUMN, "max"),
            )
            .reset_index()
            .sort_values(HIVE_COLUMN)
        )

        rows = []
        for _, row in grouped.iterrows():
            start = row["start"]
            end = row["end"]
            duration_days = None
            expected_records = None
            coverage_percentage = None
            if pd.notna(start) and pd.notna(end):
                duration_seconds = max((end - start).total_seconds(), 0)
                duration_days = round(duration_seconds / 86400, 2)
                if sampling_seconds:
                    expected_records = round(duration_seconds / sampling_seconds) + 1
                    if expected_records > 0:
                        coverage_percentage = round(
                            min(100.0, (int(row["records"]) / expected_records) * 100), 3
                        )

            rows.append(
                {
                    "hive": str(row[HIVE_COLUMN]),
                    "records": int(row["records"]),
                    "start": start.isoformat() if pd.notna(start) else None,
                    "end": end.isoformat() if pd.notna(end) else None,
                    "duration_days": duration_days,
                    "expected_records": expected_records,
                    "coverage_percentage": coverage_percentage,
                }
            )
        return rows

    @staticmethod
    def _pattern_rows(grouped: pd.DataFrame, key: str) -> list[dict[str, Any]]:
        result = []
        for _, row in grouped.iterrows():
            item = {key: int(row[key]) if key in {"hour", "day_index"} else str(row[key])}
            for source_name, api_name in SENSOR_API_NAMES.items():
                item[api_name] = EDAService._round(row[source_name])
            result.append(item)
        return result

    @staticmethod
    def _build_hourly_patterns(df: pd.DataFrame) -> list[dict[str, Any]]:
        valid = df.dropna(subset=[TIMESTAMP_COLUMN]).copy()
        valid["hour"] = valid[TIMESTAMP_COLUMN].dt.hour
        hourly = valid.groupby("hour")[list(SENSOR_COLUMNS)].mean().reset_index()
        return EDAService._pattern_rows(hourly, "hour")

    @staticmethod
    def _build_weekday_patterns(df: pd.DataFrame) -> list[dict[str, Any]]:
        valid = df.dropna(subset=[TIMESTAMP_COLUMN]).copy()
        valid["day_index"] = valid[TIMESTAMP_COLUMN].dt.dayofweek
        weekday = valid.groupby("day_index")[list(SENSOR_COLUMNS)].mean().reset_index()
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        rows = EDAService._pattern_rows(weekday, "day_index")
        for row in rows:
            row["day"] = names[row["day_index"]]
        return rows

    @staticmethod
    def _build_monthly_patterns(df: pd.DataFrame) -> list[dict[str, Any]]:
        valid = df.dropna(subset=[TIMESTAMP_COLUMN]).copy()
        valid["month"] = valid[TIMESTAMP_COLUMN].dt.to_period("M").astype(str)
        monthly = valid.groupby("month")[list(SENSOR_COLUMNS)].mean().reset_index()
        return EDAService._pattern_rows(monthly, "month")

    @staticmethod
    def _build_sensor_histograms(
        df: pd.DataFrame, bins: int = 36
    ) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for source_name, api_name in SENSOR_API_NAMES.items():
            values = pd.to_numeric(df[source_name], errors="coerce").dropna().to_numpy()
            if values.size == 0:
                result[api_name] = []
                continue
            counts, edges = np.histogram(values, bins=bins)
            result[api_name] = [
                {
                    "bin_start": EDAService._round(edges[index]),
                    "bin_end": EDAService._round(edges[index + 1]),
                    "bin_center": EDAService._round((edges[index] + edges[index + 1]) / 2),
                    "count": int(counts[index]),
                }
                for index in range(len(counts))
            ]
        return result

    @staticmethod
    def _build_target_by_hive(df: pd.DataFrame) -> list[dict[str, Any]]:
        grouped = df.groupby(HIVE_COLUMN)[list(TARGET_COLUMNS)].sum().reset_index()
        result = []
        for _, row in grouped.sort_values(HIVE_COLUMN).iterrows():
            item: dict[str, Any] = {"hive": str(row[HIVE_COLUMN])}
            for target in TARGET_COLUMNS:
                item[target] = int(row[target])
            result.append(item)
        return result

    @staticmethod
    def _build_monthly_target_counts(df: pd.DataFrame) -> list[dict[str, Any]]:
        valid = df.dropna(subset=[TIMESTAMP_COLUMN]).copy()
        valid["month"] = valid[TIMESTAMP_COLUMN].dt.to_period("M").astype(str)
        grouped = (
            valid.groupby("month")
            .agg(
                records=(TIMESTAMP_COLUMN, "size"),
                **{target: (target, "sum") for target in TARGET_COLUMNS},
            )
            .reset_index()
        )

        result = []
        for _, row in grouped.iterrows():
            item: dict[str, Any] = {
                "month": str(row["month"]),
                "records": int(row["records"]),
            }
            for target in TARGET_COLUMNS:
                count = int(row[target])
                item[target] = count
                item[f"{target}_per_10000"] = (
                    round((count / int(row["records"])) * 10000, 5) if int(row["records"]) else 0.0
                )
            result.append(item)
        return result

    @staticmethod
    def _build_target_cooccurrence(df: pd.DataFrame) -> list[dict[str, Any]]:
        binary = df[list(TARGET_COLUMNS)].fillna(0).astype(int)
        result = []
        for row_target in TARGET_COLUMNS:
            for column_target in TARGET_COLUMNS:
                count = int(((binary[row_target] == 1) & (binary[column_target] == 1)).sum())
                result.append(
                    {
                        "row": row_target,
                        "column": column_target,
                        "count": count,
                    }
                )
        return result

    @staticmethod
    def _build_target_sensor_effects(df: pd.DataFrame) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for target in TARGET_COLUMNS:
            positive_mask = df[target] == 1
            negative_mask = df[target] == 0
            effects = []
            for source_name, api_name in SENSOR_API_NAMES.items():
                series = pd.to_numeric(df[source_name], errors="coerce")
                positive_mean = series[positive_mask].mean()
                negative_mean = series[negative_mask].mean()
                overall_std = series.std()
                standardized_difference = None
                if pd.notna(overall_std) and float(overall_std) != 0:
                    standardized_difference = (positive_mean - negative_mean) / overall_std
                effects.append(
                    {
                        "sensor": api_name,
                        "positive_mean": EDAService._round(positive_mean),
                        "negative_mean": EDAService._round(negative_mean),
                        "standardized_difference": EDAService._round(
                            standardized_difference, digits=4
                        ),
                    }
                )
            result[target] = {
                "display_name": TARGET_DISPLAY_NAMES.get(target, target),
                "positive_count": int(positive_mask.sum()),
                "negative_count": int(negative_mask.sum()),
                "effects": effects,
            }
        return result

    @staticmethod
    def _build_relationship_sample(df: pd.DataFrame, limit: int = 2500) -> list[dict[str, Any]]:
        sample_columns = [TIMESTAMP_COLUMN, HIVE_COLUMN, *SENSOR_COLUMNS]
        clean = df[sample_columns].dropna().copy()
        if len(clean) > limit:
            clean = clean.sample(n=limit, random_state=42)
        clean = clean.sort_values(TIMESTAMP_COLUMN)
        result = []
        for _, row in clean.iterrows():
            result.append(
                {
                    "timestamp": row[TIMESTAMP_COLUMN].isoformat(),
                    "hive": str(row[HIVE_COLUMN]),
                    **{
                        api_name: EDAService._round(row[source_name])
                        for source_name, api_name in SENSOR_API_NAMES.items()
                    },
                }
            )
        return result

    @staticmethod
    def _build_outlier_analysis(df: pd.DataFrame) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for source_name, api_name in SENSOR_API_NAMES.items():
            series = pd.to_numeric(df[source_name], errors="coerce").dropna()
            if series.empty:
                result[api_name] = {"count": 0, "percentage": 0.0}
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)
            count = int(((series < lower) | (series > upper)).sum())
            result[api_name] = {
                "count": count,
                "percentage": round((count / len(series)) * 100, 4),
                "lower_bound": EDAService._round(lower),
                "upper_bound": EDAService._round(upper),
            }
        return result

    @staticmethod
    def _build_correlation(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
        correlation = df[list(SENSOR_COLUMNS)].corr()
        renamed = correlation.rename(index=SENSOR_API_NAMES, columns=SENSOR_API_NAMES)
        return {
            row_name: {
                column_name: EDAService._round(value, digits=4)
                for column_name, value in row.items()
            }
            for row_name, row in renamed.iterrows()
        }

    @staticmethod
    def _build_data_quality(df: pd.DataFrame) -> dict[str, Any]:
        checked_columns = [*SENSOR_COLUMNS, *TARGET_COLUMNS]
        missing_raw = {
            column: int(value) for column, value in df[checked_columns].isna().sum().items()
        }
        missing_by_column = {
            SENSOR_API_NAMES.get(column, column): count for column, count in missing_raw.items()
        }
        duplicate_timestamps = int(
            df.duplicated(subset=[HIVE_COLUMN, TIMESTAMP_COLUMN], keep=False).sum()
        )

        return {
            "missing_values": int(sum(missing_raw.values())),
            "missing_by_column": missing_by_column,
            "duplicate_timestamps": duplicate_timestamps,
            "invalid_timestamps": int(df[TIMESTAMP_COLUMN].isna().sum()),
        }

    def _generated_images(self) -> dict[str, str]:
        return {
            path.stem: f"/api/eda/images/{path.name}"
            for path in sorted(self.report_directory.glob("*.png"))
            if path.name in ALLOWED_EDA_IMAGES
        }

    @staticmethod
    def _round(value: Any, digits: int = 3) -> float | None:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
