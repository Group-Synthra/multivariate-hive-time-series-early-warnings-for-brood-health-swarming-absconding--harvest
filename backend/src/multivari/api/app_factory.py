from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from multivari.api.harvesting_live_sensor_routes import register_harvesting_live_sensor_routes
from multivari.modules.harvesting.live_hui_monitor import (
    create_live_hui_monitor,
    should_start_monitor_in_this_process,
)

from .eda_service import EDAService
from .harvesting_live_routes import create_harvesting_live_blueprint
from .routes import create_api_blueprint


def _backend_root() -> Path:
    # .../backend/src/multivari/api/app_factory.py -> .../backend
    return Path(__file__).resolve().parents[3]


def _allowed_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app() -> Flask:
    backend_root = _backend_root()

    app = Flask(__name__)
    app.config.update(
        JSON_SORT_KEYS=False,
        BACKEND_ROOT=str(backend_root),
    )

    CORS(
        app,
        resources={r"/api/*": {"origins": _allowed_origins()}},
        supports_credentials=False,
    )

    service = EDAService(backend_root=backend_root)
    app.register_blueprint(create_api_blueprint(service))

    live_hui_monitor = create_live_hui_monitor(backend_root=backend_root)
    app.extensions["live_hui_monitor"] = live_hui_monitor
    app.register_blueprint(
        create_harvesting_live_blueprint(live_hui_monitor)
    )
    if should_start_monitor_in_this_process():
        live_hui_monitor.start()

    @app.get("/")
    def index():
        base_url = request.host_url.rstrip("/")
        return jsonify(
            {
                "service": "MULTIVARI Bee Analytics API",
                "status": "running",
                "endpoints": {
                    "health": f"{base_url}/api/health",
                    "common_eda": f"{base_url}/api/eda",
                },
            }
        )

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return jsonify({"error": "HTTP method not allowed for this endpoint"}), 405

    register_harvesting_live_sensor_routes(app)

    return app
