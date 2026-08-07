from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from multivari.modules.absconding.iot_monitor import AbscondingIotMonitor
from multivari.modules.absconding.routes import create_absconding_blueprint
from multivari.modules.absconding.service import AbscondingService
from multivari.modules.brood_health.routes import create_brood_health_blueprint
from multivari.modules.brood_health.service import BroodHealthService

from .eda_service import EDAService
from .routes import create_api_blueprint


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _allowed_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    absconding_service = AbscondingService(backend_root=backend_root)
    iot_monitor = AbscondingIotMonitor(
        prediction_factory=absconding_service.build_live_iot_prediction,
        cache_path=(
            backend_root / "artifacts" / "predictions" / "absconding" / "iot_live_latest.json"
        ),
        interval_minutes=max(1, int(os.getenv("IOT_INTERVAL_MINUTES", "10"))),
        enabled=_bool_env("IOT_MONITOR_ENABLED", bool(os.getenv("DATABASE_URL"))),
    )

    app.register_blueprint(create_api_blueprint(service))
    brood_service = BroodHealthService()
    app.register_blueprint(create_brood_health_blueprint(brood_service))
    app.register_blueprint(create_absconding_blueprint(absconding_service, iot_monitor))
    app.extensions["absconding_iot_monitor"] = iot_monitor

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
                    "brood_health_eda": f"{base_url}/api/brood-health/eda",
                    "brood_health_model": f"{base_url}/api/brood-health/model",
                    "brood_health_iot": f"{base_url}/api/brood-health/iot/health",
                    "absconding": f"{base_url}/api/absconding/summary",
                    "absconding_iot": f"{base_url}/api/absconding/iot/live",
                    "absconding_iot_monitor": (f"{base_url}/api/absconding/iot/monitor/status"),
                },
            }
        )

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return jsonify({"error": "HTTP method not allowed for this endpoint"}), 405

    return app
