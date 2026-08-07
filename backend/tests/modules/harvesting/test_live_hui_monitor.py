from __future__ import annotations

import pandas as pd
import pytest

from multivari.iot.postgres_repository import LiveSensorConfigurationError
from multivari.modules.harvesting.live_hui_monitor import (
    LiveHuiMonitor,
    UnavailableLiveHuiMonitor,
)


class _Repository:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_recent(self, *, hive_id=None):
        self.calls += 1
        return pd.DataFrame({"requested_hive": [hive_id]})


class _Engine:
    def build_payload(self, raw):
        del raw
        return {
            "status": "live_classifier_derived_hui_ready",
            "available_hives": ["A", "B"],
            "latest_by_hive": [
                {"hive_id": "A", "current_hui": 20.0},
                {"hive_id": "B", "current_hui": 60.0},
            ],
            "hui_series": [
                {"hive_id": "A", "classifier_derived_hui": 20.0},
                {"hive_id": "B", "classifier_derived_hui": 60.0},
            ],
            "hive_diagnostics": [
                {"hive_id": "A", "ready_for_full_hui": True},
                {"hive_id": "B", "ready_for_full_hui": True},
            ],
        }


def test_monitor_caches_and_filters_requested_hive(tmp_path) -> None:
    repository = _Repository()
    monitor = LiveHuiMonitor(
        backend_root=tmp_path,
        enabled=False,
        interval_minutes=10,
        repository=repository,
        engine=_Engine(),
    )

    first = monitor.get_payload()
    filtered = monitor.get_payload(hive_id="B")

    assert repository.calls == 1
    assert first["available_hives"] == ["A", "B"]
    assert filtered["available_hives"] == ["B"]
    assert filtered["latest_by_hive"][0]["current_hui"] == 60.0


def test_force_refresh_calls_repository_again(tmp_path) -> None:
    repository = _Repository()
    monitor = LiveHuiMonitor(
        backend_root=tmp_path,
        enabled=False,
        interval_minutes=10,
        repository=repository,
        engine=_Engine(),
    )

    monitor.get_payload()
    monitor.get_payload(force_refresh=True)

    assert repository.calls == 2


def test_unavailable_monitor_exposes_configuration_error() -> None:
    monitor = UnavailableLiveHuiMonitor(
        LiveSensorConfigurationError("DATABASE_URL is missing")
    )

    assert monitor.status_payload()["status"] == "configuration_error"
    with pytest.raises(LiveSensorConfigurationError):
        monitor.get_payload()
