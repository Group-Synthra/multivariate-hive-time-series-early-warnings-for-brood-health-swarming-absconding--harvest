from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class AbscondingIotMonitor:
    prediction_factory: Callable[[], dict[str, Any]]
    cache_path: Path
    interval_minutes: int = 10
    enabled: bool = True
    _thread: threading.Thread | None = field(default=None, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _state: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.cache_path = Path(self.cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
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
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        with self._lock:
            self.enabled = False
            self._state["enabled"] = False
            self._state["running"] = False
        return self.status()

    def run_once(self) -> dict[str, Any]:
        started = _now()
        with self._lock:
            self._state["running"] = True
            self._state["poll_count"] += 1
            self._state["last_poll_started_at"] = started.isoformat()
            self._state["last_error"] = None
        try:
            result = self.prediction_factory()
            self.cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            finished = _now()
            with self._lock:
                self._state["running"] = False
                self._state["success_count"] += 1
                self._state["last_poll_finished_at"] = finished.isoformat()
                self._state["last_success_at"] = finished.isoformat()
            return result
        except Exception as error:
            finished = _now()
            with self._lock:
                self._state["running"] = False
                self._state["failure_count"] += 1
                self._state["last_poll_finished_at"] = finished.isoformat()
                self._state["last_error"] = str(error)
            raise

    def read_cached(self) -> dict[str, Any] | None:
        if not self.cache_path.is_file():
            return None
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        payload["api_delivery_mode"] = "backend_cached_real_iot"
        payload["backend_iot_monitor"] = self.status()
        return payload

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
        state["thread_alive"] = bool(self._thread and self._thread.is_alive())
        state["cache_exists"] = self.cache_path.is_file()
        state["cache_path"] = str(self.cache_path)
        return state

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            next_poll = _now() + timedelta(minutes=self.interval_minutes)
            with self._lock:
                self._state["next_poll_at"] = next_poll.isoformat()
            try:
                self.run_once()
            except Exception:
                pass
            self._stop_event.wait(max(60, self.interval_minutes * 60))
