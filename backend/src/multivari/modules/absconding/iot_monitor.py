from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def _now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


@dataclass
class AbscondingIotMonitor:
    """Periodically generate and cache an Absconding IoT prediction."""

    prediction_factory: Callable[[], dict[str, Any]]
    cache_path: Path
    interval_minutes: int = 10
    enabled: bool = True

    _thread: threading.Thread | None = field(default=None, init=False)
    _stop_event: threading.Event = field(
        default_factory=threading.Event,
        init=False,
    )
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
    )
    _state: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Initialise the cache directory and monitor state."""
        self.cache_path = Path(self.cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.interval_minutes = max(1, int(self.interval_minutes))

        self._state = {
            "enabled": self.enabled,
            "running": False,
            "interval_minutes": self.interval_minutes,
            "poll_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "last_poll_started_at": None,
            "last_poll_finished_at": None,
            "last_success_at": None,
            "last_error": None,
            "next_poll_at": None,
        }

    def start(self) -> dict[str, Any]:
        """Start the background IoT polling thread."""
        with self._lock:
            self.enabled = True
            self._state["enabled"] = True

            if self._thread and self._thread.is_alive():
                return self.status()

            self._stop_event.clear()

            self._thread = threading.Thread(
                target=self._loop,
                name="absconding-iot-monitor",
                daemon=True,
            )
            self._thread.start()

        LOGGER.info(
            "Absconding IoT monitor started with a %s-minute interval.",
            self.interval_minutes,
        )

        return self.status()

    def stop(self) -> dict[str, Any]:
        """Stop the background IoT polling thread."""
        self._stop_event.set()

        with self._lock:
            self.enabled = False
            self._state["enabled"] = False
            self._state["running"] = False
            self._state["next_poll_at"] = None

        LOGGER.info("Absconding IoT monitor stopped.")

        return self.status()

    def run_once(self) -> dict[str, Any]:
        """Run one database retrieval and Absconding prediction cycle."""
        started = _now()

        with self._lock:
            self._state["running"] = True
            self._state["poll_count"] += 1
            self._state["last_poll_started_at"] = started.isoformat()
            self._state["last_error"] = None

        try:
            result = self.prediction_factory()

            temporary_cache_path = self.cache_path.with_suffix(
                f"{self.cache_path.suffix}.tmp"
            )
            temporary_cache_path.write_text(
                json.dumps(result, indent=2, default=str),
                encoding="utf-8",
            )
            temporary_cache_path.replace(self.cache_path)

            finished = _now()

            with self._lock:
                self._state["running"] = False
                self._state["success_count"] += 1
                self._state["last_poll_finished_at"] = finished.isoformat()
                self._state["last_success_at"] = finished.isoformat()

            LOGGER.info(
                "Absconding IoT prediction completed successfully at %s.",
                finished.isoformat(),
            )

            return result

        except Exception as error:
            finished = _now()

            with self._lock:
                self._state["running"] = False
                self._state["failure_count"] += 1
                self._state["last_poll_finished_at"] = finished.isoformat()
                self._state["last_error"] = str(error)

            LOGGER.exception(
                "Absconding IoT prediction cycle failed at %s.",
                finished.isoformat(),
            )

            raise

    def read_cached(self) -> dict[str, Any] | None:
        """Read the most recently cached IoT prediction."""
        if not self.cache_path.is_file():
            return None

        try:
            payload = json.loads(
                self.cache_path.read_text(encoding="utf-8"),
            )
        except (OSError, json.JSONDecodeError) as error:
            LOGGER.warning(
                "Unable to read the cached Absconding IoT prediction: %s",
                error,
            )
            return None

        if not isinstance(payload, dict):
            LOGGER.warning("Cached Absconding IoT prediction is not a JSON object.")
            return None

        payload["api_delivery_mode"] = "backend_cached_real_iot"
        payload["backend_iot_monitor"] = self.status()

        return payload

    def status(self) -> dict[str, Any]:
        """Return the current monitor status."""
        with self._lock:
            state = dict(self._state)

        state["thread_alive"] = bool(self._thread and self._thread.is_alive())
        state["cache_exists"] = self.cache_path.is_file()
        state["cache_path"] = str(self.cache_path)

        return state

    def _loop(self) -> None:
        """Continuously run IoT prediction cycles until stopped."""
        while not self._stop_event.is_set():
            next_poll = _now() + timedelta(
                minutes=self.interval_minutes,
            )

            with self._lock:
                self._state["next_poll_at"] = next_poll.isoformat()

            try:
                self.run_once()
            except Exception:
                LOGGER.exception(
                    "Absconding IoT polling cycle failed; retrying at the next interval."
                )

            wait_seconds = max(
                60,
                self.interval_minutes * 60,
            )
            self._stop_event.wait(wait_seconds)

        with self._lock:
            self._state["running"] = False
            self._state["next_poll_at"] = None

        LOGGER.info("Absconding IoT monitor background loop exited.")