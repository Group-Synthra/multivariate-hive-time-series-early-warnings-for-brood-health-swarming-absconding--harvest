import json
import os
from pathlib import Path

import pandas as pd
from flask import Blueprint, jsonify, send_file

model_training_bp = Blueprint("model_training", __name__)

BACKEND_DIR = Path(__file__).resolve().parents[4]
BASE_PATH = BACKEND_DIR / "artifacts" / "metrics" / "swarming"
CHART_PATH = BACKEND_DIR / "artifacts" / "reports" / "swarming" / "training"


def load_json(filename):
    path = BASE_PATH / filename
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


@model_training_bp.route("/api/swarming/model-training", methods=["GET"])
def model_training():
    rf = load_json("rf_metrics.json")
    xgb = load_json("xgb_metrics.json")
    lstm = load_json("lstm_metrics.json")
    best = load_json("best_model.json")
    pelt = load_json("pelt_metrics.json")
    comparison_path = os.path.join(str(BASE_PATH), "model_comparison.csv")

    comparison = []
    if os.path.exists(comparison_path):
        comparison = pd.read_csv(comparison_path).to_dict(orient="records")

    return jsonify(
        {
            "pelt": pelt,
            "rf": rf,
            "xgb": xgb,
            "lstm": lstm,
            "best_model": best,
            "comparison": comparison,
        }
    )


@model_training_bp.route("/api/swarming/model-training/chart")
def comparison_chart():
    path = os.path.join(str(CHART_PATH), "model_comparison.png")
    return send_file(path, mimetype="image/png")
