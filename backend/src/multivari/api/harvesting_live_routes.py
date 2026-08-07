from __future__ import annotations

from flask import Blueprint, jsonify, request

from multivari.iot.postgres_repository import (
    LiveSensorConfigurationError,
    LiveSensorDatabaseError,
)
from multivari.modules.harvesting.live_hui_inference import (
    InsufficientLiveHistoryError,
    LiveHuiArtifactError,
    LiveHuiInferenceError,
)
from multivari.modules.harvesting.live_hui_monitor import LiveHuiMonitor


def create_harvesting_live_blueprint(monitor: LiveHuiMonitor) -> Blueprint:
    api = Blueprint(
        "harvesting_live",
        __name__,
        url_prefix="/api/harvesting",
    )

    @api.get("/live-hui")
    def live_hui():
        hive_id = request.args.get("hive_id", default=None, type=str)
        force_refresh = request.args.get("refresh", default="false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            payload = monitor.get_payload(
                hive_id=hive_id.strip() if hive_id else None,
                force_refresh=force_refresh,
            )
            status_code = 200 if payload.get("latest_by_hive") else 422
            return jsonify(payload), status_code
        except InsufficientLiveHistoryError as error:
            return (
                jsonify(
                    {
                        "status": "insufficient_live_history",
                        "error": str(error),
                        "diagnostics": error.diagnostics,
                    }
                ),
                422,
            )
        except (LiveSensorConfigurationError, LiveHuiArtifactError) as error:
            return jsonify({"status": "configuration_error", "error": str(error)}), 503
        except LiveSensorDatabaseError as error:
            return jsonify({"status": "database_unavailable", "error": str(error)}), 503
        except LiveHuiInferenceError as error:
            return jsonify({"status": "inference_error", "error": str(error)}), 500

    @api.post("/live-hui/refresh")
    def refresh_live_hui():
        body = request.get_json(silent=True) or {}
        hive_id = body.get("hive_id")
        try:
            payload = monitor.refresh(hive_id=str(hive_id).strip() if hive_id else None)
            return jsonify(payload)
        except InsufficientLiveHistoryError as error:
            return (
                jsonify(
                    {
                        "status": "insufficient_live_history",
                        "error": str(error),
                        "diagnostics": error.diagnostics,
                    }
                ),
                422,
            )
        except (LiveSensorConfigurationError, LiveHuiArtifactError) as error:
            return jsonify({"status": "configuration_error", "error": str(error)}), 503
        except LiveSensorDatabaseError as error:
            return jsonify({"status": "database_unavailable", "error": str(error)}), 503
        except LiveHuiInferenceError as error:
            return jsonify({"status": "inference_error", "error": str(error)}), 500

    @api.get("/live-hui/status")
    def live_hui_status():
        return jsonify(monitor.status_payload())

    return api
