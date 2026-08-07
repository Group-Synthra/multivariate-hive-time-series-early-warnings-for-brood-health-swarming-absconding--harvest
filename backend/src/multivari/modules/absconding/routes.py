from __future__ import annotations

from flask import Blueprint, jsonify, request, send_file

from .iot_monitor import AbscondingIotMonitor
from .service import AbscondingService


def create_absconding_blueprint(
    service: AbscondingService,
    monitor: AbscondingIotMonitor | None = None,
) -> Blueprint:
    blueprint = Blueprint("absconding", __name__, url_prefix="/api/absconding")

    @blueprint.get("/summary")
    def summary():
        try:
            return jsonify(service.get_dashboard())
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 503

    @blueprint.get("/metrics")
    def metrics():
        try:
            dashboard = service.get_dashboard()
            return jsonify(dashboard.get("model_training", {}))
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 503

    @blueprint.get("/hives")
    def hives():
        try:
            dashboard = service.get_dashboard()
            return jsonify(
                {
                    "hive_options": dashboard.get("hive_options", []),
                    "latest_hive_risk": dashboard.get("latest_hive_risk", []),
                }
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 503

    @blueprint.get("/hives/<path:hive_id>")
    def hive_detail(hive_id: str):
        try:
            return jsonify(service.get_hive(hive_id))
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 503
        except KeyError:
            return jsonify({"error": f"Unknown hive: {hive_id}"}), 404

    @blueprint.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(service.predict(payload))
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 503
        except (ValueError, TypeError, KeyError) as error:
            return jsonify({"error": str(error)}), 400

    @blueprint.get("/iot/live")
    def iot_live():
        if monitor is None:
            return jsonify({"error": "The Absconding IoT monitor is not configured."}), 503
        force = request.args.get("force", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            if force:
                result = monitor.run_once()
                result["api_delivery_mode"] = "forced_fresh_database_read"
                result["backend_iot_monitor"] = monitor.status()
                return jsonify(result)

            cached = monitor.read_cached()
            if cached is not None:
                return jsonify(cached)

            result = monitor.run_once()
            result["api_delivery_mode"] = "initial_fresh_database_read"
            result["backend_iot_monitor"] = monitor.status()
            return jsonify(result)
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
            cached = monitor.read_cached()
            if cached is not None:
                cached["live_warning"] = str(error)
                cached["backend_iot_monitor"] = monitor.status()
                return jsonify(cached), 200
            return jsonify({"error": str(error), "monitor": monitor.status()}), 503

    @blueprint.get("/iot/monitor/status")
    def iot_monitor_status():
        if monitor is None:
            return jsonify({"configured": False, "error": "IoT monitor is unavailable."}), 503
        return jsonify({"configured": True, **monitor.status()})

    @blueprint.post("/iot/monitor/start")
    def iot_monitor_start():
        if monitor is None:
            return jsonify({"error": "IoT monitor is unavailable."}), 503
        return jsonify(monitor.start())

    @blueprint.post("/iot/monitor/stop")
    def iot_monitor_stop():
        if monitor is None:
            return jsonify({"error": "IoT monitor is unavailable."}), 503
        return jsonify(monitor.stop())

    @blueprint.post("/iot/monitor/run-now")
    def iot_monitor_run_now():
        if monitor is None:
            return jsonify({"error": "IoT monitor is unavailable."}), 503
        try:
            result = monitor.run_once()
            return jsonify({"prediction": result, "monitor": monitor.status()})
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
            return jsonify({"error": str(error), "monitor": monitor.status()}), 503

    @blueprint.get("/images/<path:filename>")
    def image(filename: str):
        try:
            return send_file(service.image_path(filename), conditional=True)
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    return blueprint
