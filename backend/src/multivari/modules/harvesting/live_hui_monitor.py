from __future__ import annotations

import logging
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from multivari.iot.postgres_repository import (
    LiveSensorConfigurationError,
    PostgresSensorRepository,
    PostgresSensorSettings,
)
from multivari.modules.harvesting.live_hui_inference import (
    InsufficientLiveHistoryError,
    LiveHuiArtifactError,
    LiveHuiArtifactSettings,
    LiveHuiInferenceEngine,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class UnavailableLiveHuiMonitor:
    """Keep the API available when live configuration is incomplete."""

    def __init__(self, error: Exception) -> None:
        self.enabled = False
        self.interval_minutes = 0
        self.error = str(error)

    def refresh(self, *, hive_id: str | None = None) -> dict[str, Any]:
        del hive_id
        raise LiveSensorConfigurationError(self.error)

    def get_payload(
        self,
        *,
        hive_id: str | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        del hive_id, force_refresh
        raise LiveSensorConfigurationError(self.error)

    def status_payload(self) -> dict[str, Any]:
        return {
            "status": "configuration_error",
            "enabled": False,
            "thread_running": False,
            "last_error": self.error,
            "cached_hives": [],
        }

    def start(self) -> None:
        return

    def stop(self) -> None:
        return


class LiveHuiMonitor:
    """Cache and periodically refresh live HUI predictions."""

    def __init__(
        self,
        *,
        backend_root: str | Path,
        enabled: bool,
        interval_minutes: int,
        repository: PostgresSensorRepository,
        engine: LiveHuiInferenceEngine,
    ) -> None:
        if interval_minutes <= 0:
            raise ValueError("IOT_INTERVAL_MINUTES must be positive.")
        self.backend_root = Path(backend_root).resolve()
        self.enabled = enabled
        self.interval_minutes = interval_minutes
        self.repository = repository
        self.engine = engine
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._payload: dict[str, Any] | None = None
        self._last_attempt_at: datetime | None = None
        self._last_success_at: datetime | None = None
        self._last_error: str | None = None

    def refresh(self, *, hive_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            self._last_attempt_at = datetime.now(UTC)
            try:
                raw = self.repository.fetch_recent(hive_id=hive_id)
                payload = self.engine.build_payload(raw)
            except Exception as error:
                self._last_error = str(error)
                raise
            self._payload = payload
            self._last_success_at = datetime.now(UTC)
            self._last_error = None
            return payload

    @staticmethod
    def _filter_payload(payload: dict[str, Any], hive_id: str | None) -> dict[str, Any]:
        if not hive_id:
            return payload
        latest = [
            row for row in payload.get("latest_by_hive", []) if str(row.get("hive_id")) == hive_id
        ]
        series = [
            row for row in payload.get("hui_series", []) if str(row.get("hive_id")) == hive_id
        ]
        diagnostics = [
            row for row in payload.get("hive_diagnostics", []) if str(row.get("hive_id")) == hive_id
        ]
        output = dict(payload)
        output["available_hives"] = [hive_id] if latest else []
        output["latest_by_hive"] = latest
        output["hui_series"] = series
        output["hive_diagnostics"] = diagnostics
        if not latest:
            output["status"] = "requested_hive_not_ready"
        return output

    def get_payload(
        self,
        *,
        hive_id: str | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            stale = self._last_success_at is None or datetime.now(
                UTC
            ) - self._last_success_at >= timedelta(minutes=self.interval_minutes)
            if force_refresh or self._payload is None or stale:
                payload = self.refresh(hive_id=None)
            else:
                payload = self._payload
        return self._filter_payload(payload, hive_id)

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok" if self._last_error is None else "degraded",
                "enabled": self.enabled,
                "interval_minutes": self.interval_minutes,
                "thread_running": bool(self._thread and self._thread.is_alive()),
                "last_attempt_at": (
                    self._last_attempt_at.isoformat() if self._last_attempt_at else None
                ),
                "last_success_at": (
                    self._last_success_at.isoformat() if self._last_success_at else None
                ),
                "last_error": self._last_error,
                "cached_hives": (self._payload.get("available_hives", []) if self._payload else []),
            }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.refresh()
            except InsufficientLiveHistoryError as error:
                logging.getLogger(__name__).info(
                    "Live HUI is collecting contiguous history: %s",
                    error,
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "Live HUI monitor refresh failed; the monitor will retry "
                    "after the configured interval."
                )
            self._stop_event.wait(self.interval_minutes * 60)

    def start(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="live-hui-monitor",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()


def create_live_hui_monitor(
    *, backend_root: str | Path
) -> LiveHuiMonitor | UnavailableLiveHuiMonitor:
    try:
        sensor_settings = PostgresSensorSettings.from_env()
        artifact_settings = LiveHuiArtifactSettings.from_env(backend_root=Path(backend_root))
        repository = PostgresSensorRepository(sensor_settings)
        engine = LiveHuiInferenceEngine(
            backend_root=backend_root,
            sensor_settings=sensor_settings,
            artifact_settings=artifact_settings,
        )
    except (LiveSensorConfigurationError, LiveHuiArtifactError) as error:
        return UnavailableLiveHuiMonitor(error)

    return LiveHuiMonitor(
        backend_root=backend_root,
        enabled=_env_bool("IOT_MONITOR_ENABLED", True),
        interval_minutes=int(os.getenv("IOT_INTERVAL_MINUTES", "10")),
        repository=repository,
        engine=engine,
    )


def should_start_monitor_in_this_process() -> bool:
    if not _env_bool("IOT_MONITOR_ENABLED", True):
        return False
    debug_enabled = _env_bool("FLASK_DEBUG", True)
    if not debug_enabled:
        return True
    # Werkzeug starts a parent and a child process in debug mode.
    return os.getenv("WERKZEUG_RUN_MAIN", "").lower() == "true"


def create_disabled_monitor_payload(error: Exception) -> dict[str, Any]:
    return {
        "status": "configuration_error",
        "error": str(error),
    }


__all__ = [
    "LiveHuiMonitor",
    "LiveSensorConfigurationError",
    "create_live_hui_monitor",
    "should_start_monitor_in_this_process",
]
