from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, jsonify, request, send_from_directory

from .config import PATHS
from .predictor import ModelNotReadyError
from .repository import IoTConfigurationError, IoTRepositoryError
from .service import BroodHealthService


def create_brood_health_blueprint(service: BroodHealthService | None = None) -> Blueprint:
    module_service = service or BroodHealthService()
    blueprint = Blueprint("brood_health", __name__, url_prefix="/api/brood-health")

    def error_response(exc: Exception, status: int = 400):
        return jsonify({"error": str(exc), "type": exc.__class__.__name__}), status

    @blueprint.get("/health")
    def health():
        return jsonify(module_service.iot_health())

    @blueprint.get("/eda")
    def eda():
        try:
            force = request.args.get("force", "false").lower() in {"1", "true", "yes"}
            return jsonify(module_service.get_eda(force=force))
        # API boundary: convert unexpected EDA failures into a JSON 500 response.
        except Exception as exc:  # noqa: BLE001
            return error_response(exc, 500)

    @blueprint.get("/model")
    def model_summary():
        return jsonify(module_service.get_model_summary())

    @blueprint.post("/train")
    def train():
        payload = request.get_json(silent=True) or {}
        try:
            horizon = int(payload.get("horizon_hours", 6))
            fast_mode = bool(payload.get("fast_mode", False))
            if not 1 <= horizon <= 24:
                raise ValueError("horizon_hours must be between 1 and 24")
            return jsonify(
                module_service.start_training(horizon_hours=horizon, fast_mode=fast_mode)
            ), 202
        # API boundary: preserve the existing JSON error contract for bad requests.
        except Exception as exc:  # noqa: BLE001
            return error_response(exc)

    @blueprint.get("/train/status")
    def training_status():
        return jsonify(module_service.training_status())

    @blueprint.get("/iot/health")
    def iot_health():
        return jsonify(module_service.iot_health())

    @blueprint.get("/iot/devices")
    def devices():
        try:
            return jsonify(module_service.list_devices())
        except (IoTConfigurationError, IoTRepositoryError) as exc:
            return error_response(exc, 503)

    @blueprint.get("/iot/predict")
    def predict_device():
        device_id = request.args.get("device_id", "").strip()
        try:
            if not device_id:
                raise ValueError("device_id query parameter is required")
            lookback = request.args.get("lookback_hours")
            lookback_hours = int(lookback) if lookback else None
            return jsonify(module_service.predict_device(device_id, lookback_hours=lookback_hours))
        except ModelNotReadyError as exc:
            return error_response(exc, 503)
        except (IoTConfigurationError, IoTRepositoryError) as exc:
            return error_response(exc, 503)
        # API boundary: prediction validation errors must remain JSON responses.
        except Exception as exc:  # noqa: BLE001
            return error_response(exc)

    @blueprint.get("/iot/validation-log")
    def validation_log():
        try:
            device_id = request.args.get("device_id", "").strip() or None
            limit = int(request.args.get("limit", "100"))
            return jsonify(
                module_service.validation_summary(
                    device_id=device_id,
                    limit=max(1, min(limit, 500)),
                )
            )
        except (IoTConfigurationError, IoTRepositoryError) as exc:
            return error_response(exc, 503)
        except Exception as exc:  # noqa: BLE001
            return error_response(exc)

    @blueprint.get("/iot/validation-log/download")
    def download_validation_log():
        device_id = request.args.get("device_id", "").strip() or None
        try:
            csv_text = module_service.validation_csv(device_id=device_id)
        except (IoTConfigurationError, IoTRepositoryError) as exc:
            return error_response(exc, 503)
        except Exception as exc:  # noqa: BLE001
            return error_response(exc)

        suffix = f"_{device_id}" if device_id else ""
        filename = f"brood_health_forecast_validation{suffix}.csv"
        return Response(
            csv_text,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )

    @blueprint.get("/iot/predict-all")
    def predict_all():
        try:
            lookback = request.args.get("lookback_hours")
            lookback_hours = int(lookback) if lookback else None
            return jsonify(module_service.predict_all_devices(lookback_hours=lookback_hours))
        except ModelNotReadyError as exc:
            return error_response(exc, 503)
        except (IoTConfigurationError, IoTRepositoryError) as exc:
            return error_response(exc, 503)

    @blueprint.post("/predict/manual")
    def manual_prediction():
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(module_service.predict_manual(payload.get("readings", [])))
        except ModelNotReadyError as exc:
            return error_response(exc, 503)
        # API boundary: malformed manual readings must remain JSON responses.
        except Exception as exc:  # noqa: BLE001
            return error_response(exc)

    @blueprint.get("/reports/<path:filename>")
    def report_image(filename: str):
        # Only generated PNG files inside brood-health report subdirectories are exposed.
        if not filename.lower().endswith(".png") or ".." in Path(filename).parts:
            return error_response(ValueError("Invalid report filename"), 404)
        for directory in (PATHS.report_dir / "eda", PATHS.report_dir / "model"):
            candidate = directory / filename
            if candidate.is_file():
                return send_from_directory(directory, filename)
        return error_response(FileNotFoundError("Report image not found"), 404)

    return blueprint
