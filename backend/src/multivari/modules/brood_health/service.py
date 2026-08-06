from __future__ import annotations

import json
import threading
import traceback
from typing import Any

import pandas as pd

from .config import PATHS, IoTSettings
from .eda import build_brood_eda
from .predictor import BroodHealthPredictor, ModelNotReadyError
from .repository import IoTConfigurationError, IoTRepositoryError, PostgresIoTRepository
from .training import run_training


class BroodHealthService:
    def __init__(self) -> None:
        self.repository = PostgresIoTRepository(IoTSettings.from_environment())
        self.predictor = BroodHealthPredictor()
        self._eda_cache: dict[str, Any] | None = None
        self._eda_source_mtime: float | None = None
        self._training_lock = threading.Lock()
        self._training_state: dict[str, Any] = {
            "running": False,
            "progress": 0,
            "event": "idle",
            "message": "Training has not been started from this server process.",
            "model": None,
            "error": None,
        }

    def get_eda(self, *, force: bool = False) -> dict[str, Any]:
        source = PATHS.clean_data if PATHS.clean_data.exists() else PATHS.raw_workbook
        source_mtime = source.stat().st_mtime if source.exists() else None
        if (
            not force
            and self._eda_cache is not None
            and self._eda_source_mtime == source_mtime
        ):
            return self._eda_cache
        if (
            not force
            and PATHS.eda_cache.exists()
            and source_mtime is not None
            and PATHS.eda_cache.stat().st_mtime >= source_mtime
        ):
            self._eda_cache = json.loads(
                PATHS.eda_cache.read_text(encoding="utf-8")
            )
            self._eda_source_mtime = source_mtime
            return self._eda_cache

        self._eda_cache = build_brood_eda(save_cache=True)
        self._eda_source_mtime = source_mtime
        return self._eda_cache

    def get_model_summary(self) -> dict[str, Any]:
        if not PATHS.training_summary.exists():
            return {
                "trained": False,
                "message": (
                    "No Brood Health v4 model was found. Run the cleanup script and "
                    "train the exact +6-hour multi-horizon forecaster."
                ),
                "training_status": self.training_status(),
            }
        summary = json.loads(PATHS.training_summary.read_text(encoding="utf-8"))
        summary["training_status"] = self.training_status()
        return summary

    def training_status(self) -> dict[str, Any]:
        with self._training_lock:
            return dict(self._training_state)

    def _progress_callback(self, event: str, payload: dict[str, Any]) -> None:
        with self._training_lock:
            self._training_state.update(
                {
                    "running": event != "complete",
                    "event": event,
                    "progress": int(
                        payload.get(
                            "progress",
                            self._training_state.get("progress", 0),
                        )
                    ),
                    "message": payload.get(
                        "message", self._training_state.get("message")
                    ),
                    "model": payload.get("model"),
                    "error": None,
                }
            )

    def start_training(
        self,
        *,
        horizon_hours: int = 6,
        fast_mode: bool = False,
    ) -> dict[str, Any]:
        with self._training_lock:
            if self._training_state.get("running"):
                return dict(self._training_state)
            self._training_state = {
                "running": True,
                "progress": 1,
                "event": "queued",
                "message": "Brood Health v4 training has been queued.",
                "model": None,
                "error": None,
                "horizon_hours": int(horizon_hours),
                "fast_mode": bool(fast_mode),
            }

        def worker() -> None:
            try:
                run_training(
                    horizon_hours=int(horizon_hours),
                    fast_mode=bool(fast_mode),
                    progress_callback=self._progress_callback,
                )
                with self._training_lock:
                    self._training_state.update(
                        {
                            "running": False,
                            "progress": 100,
                            "event": "complete",
                            "message": "Training completed successfully.",
                            "error": None,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                with self._training_lock:
                    self._training_state.update(
                        {
                            "running": False,
                            "event": "failed",
                            "message": "Training failed.",
                            "error": str(exc),
                            "traceback": traceback.format_exc(limit=8),
                        }
                    )

        threading.Thread(
            target=worker,
            name="brood-health-training-v4",
            daemon=True,
        ).start()
        return self.training_status()

    def iot_health(self) -> dict[str, Any]:
        try:
            database = self.repository.health()
        except (IoTConfigurationError, IoTRepositoryError) as exc:
            database = {
                "configured": bool(self.repository.settings.database_url),
                "connected": False,
                "error": str(exc),
                "schema": self.repository.settings.schema,
                "table": self.repository.settings.table,
            }
        try:
            model = {"ready": True, **self.predictor.model_info()}
        except ModelNotReadyError as exc:
            model = {"ready": False, "error": str(exc)}
        return {"database": database, "model": model}

    def list_devices(self) -> dict[str, Any]:
        return {
            "devices": self.repository.list_devices(),
            "lookback_hours": self.repository.settings.lookback_hours,
            "refresh_seconds": self.repository.settings.refresh_seconds,
        }

    def _predict_frame(self, history: pd.DataFrame) -> dict[str, Any]:
        settings = self.repository.settings
        return self.predictor.predict_raw_iot(
            history,
            weight_scale_factor=settings.weight_scale_factor,
            weight_offset_kg=settings.weight_offset_kg,
        )

    def predict_device(
        self,
        device_id: str,
        *,
        lookback_hours: int | None = None,
    ) -> dict[str, Any]:
        history = self.repository.fetch_history(
            device_id,
            lookback_hours=lookback_hours,
        )
        if history.empty:
            raise ValueError(
                f"No IoT readings were found for device {device_id}"
            )
        prediction = self._predict_frame(history)
        if (
            int(prediction.get("hourly_rows", 0))
            < self.repository.settings.minimum_hourly_rows
        ):
            raise ValueError(
                f"At least {self.repository.settings.minimum_hourly_rows} complete "
                f"hourly rows are required; received "
                f"{prediction.get('hourly_rows', 0)}"
            )
        prediction["source"] = "postgresql"
        prediction["raw_rows"] = len(history)
        prediction["database_weight_conversion"] = {
            "scale_factor": self.repository.settings.weight_scale_factor,
            "offset_kg": self.repository.settings.weight_offset_kg,
        }
        return prediction

    def predict_all_devices(
        self,
        *,
        lookback_hours: int | None = None,
    ) -> dict[str, Any]:
        devices = self.repository.list_devices()
        predictions: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for item in devices:
            device_id = item["device_id"]
            try:
                prediction = self.predict_device(
                    device_id,
                    lookback_hours=lookback_hours,
                )
                predictions.append(
                    {
                        "device_id": device_id,
                        "latest_timestamp": prediction["latest_timestamp"],
                        "exact_score": prediction["prediction"]["exact_score"],
                        "exact_level": prediction["prediction"]["exact_level"],
                        "safety_minimum_score": prediction["prediction"][
                            "safety_minimum_score"
                        ],
                        "warning_level": prediction["warning"]["level"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"device_id": device_id, "error": str(exc)})
        return {"predictions": predictions, "failures": failures}

    def predict_manual(self, readings: list[dict[str, Any]]) -> dict[str, Any]:
        if not readings:
            raise ValueError("readings must be a non-empty array")
        prediction = self._predict_frame(pd.DataFrame(readings))
        if (
            int(prediction.get("hourly_rows", 0))
            < self.repository.settings.minimum_hourly_rows
        ):
            raise ValueError(
                f"At least {self.repository.settings.minimum_hourly_rows} complete "
                f"hourly rows are required; received "
                f"{prediction.get('hourly_rows', 0)}"
            )
        prediction["source"] = "manual_payload"
        prediction["raw_rows"] = len(readings)
        return prediction
