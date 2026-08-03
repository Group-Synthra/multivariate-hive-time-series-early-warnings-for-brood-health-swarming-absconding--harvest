from __future__ import annotations

from flask import Blueprint, jsonify, send_file

from .eda_service import EDAService


def create_api_blueprint(service: EDAService) -> Blueprint:
    api = Blueprint("api", __name__, url_prefix="/api")

    @api.get("/health")
    def health():
        return jsonify(service.health_payload())

    @api.get("/eda")
    def common_eda():
        try:
            return jsonify(service.get_common_eda())
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 503
        except (KeyError, ValueError, TypeError) as error:
            return jsonify({"error": f"Unable to build EDA response: {error}"}), 500

    @api.get("/eda/images/<path:filename>")
    def common_eda_image(filename: str):
        try:
            return send_file(service.image_path(filename), conditional=True)
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404

    return api
