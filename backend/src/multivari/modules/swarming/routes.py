"""
=========================================================
Swarming Live Prediction API — Flask Blueprint
=========================================================

Routes:
  POST  /api/swarming/live-prediction
        Accepts: { "hive_id": str, "readings": [ ...24 dicts... ] }
        Returns: prediction JSON

  GET   /api/swarming/live-prediction/sample
        Returns a sample request payload for testing.

  GET   /api/swarming/live-prediction/health
        Returns model availability status.

  GET   /api/swarming/predict-from-iot?device_id=<id>&limit=<n>
        Fetches last N real IoT readings from Supabase, runs LSTM,
        returns: latest_sensor + prediction + 3-day forecast.
=========================================================
"""

import json
import logging
import os
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request, send_from_directory
from sqlalchemy import text

from .history_backfill import backfill_missing_risks
from .iot.database import get_engine
from .risk_history_json import get_risk_history, save_risk_prediction

logger = logging.getLogger(__name__)

swarming_live_bp = Blueprint("swarming_live", __name__)

_BACKEND_DIR = Path(__file__).resolve().parents[4]
_EDA_DIR = _BACKEND_DIR / "artifacts" / "reports" / "swarming" / "eda"

# Lazy import to avoid crashing Flask startup if TF is slow to load
_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from .live_prediction import live_predictor

        _predictor = live_predictor
    return _predictor


# -------------------------------------------------------
# Sample payload — realistic values for a high-risk hive
# -------------------------------------------------------


def _generate_sample_readings(n: int = 144) -> list:
    """Generate synthetic readings that mimic a pre-swarm state."""
    random.seed(42)
    readings = []
    for i in range(n):
        # Gradual temperature rise and CO2 spike toward the end
        temp_trend = 0.3 * i
        co2_trend = 15 * i
        weight_drop = -0.05 * i
        readings.append(
            {
                "recorded_at": (
                    datetime.now(UTC) - timedelta(minutes=10 * (n - 1 - i))
                ).isoformat(),
                "internal_temperature_c": round(34.5 + temp_trend + random.uniform(-0.2, 0.2), 2),
                "internal_humidity_pct": round(65.0 + random.uniform(-1.0, 1.0), 2),
                "co2_ppm": round(1200 + co2_trend + random.uniform(-30, 30), 2),
                "hive_weight_kg": round(35.0 + weight_drop + random.uniform(-0.1, 0.1), 2),
                "external_temperature_c": round(28.0 + random.uniform(-0.5, 0.5), 2),
                "external_humidity_pct": round(55.0 + random.uniform(-1.0, 1.0), 2),
                "rainfall_mm_hour": round(max(0, random.uniform(0, 0.2)), 2),
                "wind_speed_mps": round(2.0 + random.uniform(0, 0.5), 2),
            }
        )
    return readings


# ──────────────────────────────────────────────────────────────
# POST /api/swarming/live-prediction
# ──────────────────────────────────────────────────────────────
@swarming_live_bp.route("/api/swarming/live-prediction", methods=["POST"])
def live_prediction():
    """
    Run live swarming prediction.

    Request body (JSON):
    {
        "hive_id" : "Hive_01",
        "readings": [
            {
                "internal_temperature_c" : 35.2,
                "internal_humidity_pct"  : 64.5,
                "co2_ppm"                : 1850,
                "hive_weight_kg"         : 32.1,
                "external_temperature_c" : 27.0,
                "external_humidity_pct"  : 55.3,
                "rainfall_mm_hour"       : 0.0,
                "wind_speed_mps"         : 2.1
            },
            ...  (24 entries total)
        ]
    }
    """
    # ── Validate Content-Type ─────────────────────────────────
    if not request.is_json:
        return jsonify(
            {"error": "Request must be JSON.", "hint": "Set Content-Type: application/json"}
        ), 415

    body = request.get_json(silent=True)

    if body is None:
        return jsonify({"error": "Invalid or empty JSON body."}), 400

    # ── Validate required fields ──────────────────────────────
    hive_id = body.get("hive_id", "Hive_Unknown")
    readings = body.get("readings", None)

    if readings is None:
        return jsonify(
            {
                "error": "Missing 'readings' field.",
                "hint": "Provide a list of sensor reading dicts.",
            }
        ), 400

    if not isinstance(readings, list):
        return jsonify(
            {
                "error": "'readings' must be a list.",
            }
        ), 400

    if len(readings) < 24:
        return jsonify(
            {
                "error": f"Insufficient data: {len(readings)} readings provided.",
                "required": 24,
                "received": len(readings),
                "hint": "Send at least 24 consecutive sensor readings.",
            }
        ), 422

    # ── Run prediction ────────────────────────────────────────
    try:
        predictor = _get_predictor()
        result = predictor.predict(hive_id, readings)

        # ── Enhance with RiskClassifier ──────────────────────
        from .live_prediction.risk_classifier import create_risk_classifier

        classifier = create_risk_classifier()
        combination = classifier.combine_lstm_and_pelt(
            lstm_probability=result.get("probability", 0),
            pelt_snapshot=result.get("pelt_snapshot", {}),
        )
        classification = classifier.classify_from_probability(combination["combined_probability"])

        # Add risk classification to result
        result.update(combination)
        result["combined_probability"] = combination["combined_probability"]
        result["risk_percentage"] = combination["risk_percentage"]
        result["risk_level"] = classification["risk_level"]
        result["warning"] = classification["message"]
        result["event_window"] = classification["event_window"]
        result["softmax_probabilities"] = classification.get("softmax_probabilities", {})
        result["predicted_class"] = (
            "Swarming"
            if combination["combined_probability"] >= result.get("threshold_used", 0.70)
            else "No Swarming"
        )

        return jsonify(result), 200

    except ValueError as e:
        logger.warning("Validation error in live prediction: %s", e)
        return jsonify({"error": "Input validation failed.", "details": str(e)}), 422

    except FileNotFoundError as e:
        logger.error("Model file missing: %s", e)
        return jsonify(
            {
                "error": "Model files not found.",
                "details": str(e),
                "hint": "Run the LSTM training pipeline to generate model files.",
            }
        ), 503

    except Exception as e:
        logger.exception("Unexpected error during prediction.")
        return jsonify(
            {"error": "Prediction failed due to an internal error.", "details": str(e)}
        ), 500


# ──────────────────────────────────────────────────────────────
# GET /api/swarming/live-prediction/sample
# ──────────────────────────────────────────────────────────────
@swarming_live_bp.route("/api/swarming/live-prediction/sample", methods=["GET"])
def sample_payload():
    """
    Return a sample request payload that can be POSTed to /live-prediction.
    Useful for testing without a real sensor.
    """
    sample = {"hive_id": "Hive_01", "readings": _generate_sample_readings(144)}
    return jsonify(sample), 200


# ──────────────────────────────────────────────────────────────
# GET /api/swarming/live-prediction/health
# ──────────────────────────────────────────────────────────────
@swarming_live_bp.route("/api/swarming/live-prediction/health", methods=["GET"])
def prediction_health():
    """Return availability status of the model files."""
    from .live_prediction.config import LABEL_ENCODER_PATH, LSTM_MODEL_PATH, SCALER_PATH

    status = {
        "lstm_model_ready": os.path.exists(LSTM_MODEL_PATH),
        "scaler_ready": os.path.exists(SCALER_PATH),
        "label_encoder_ready": os.path.exists(LABEL_ENCODER_PATH),
    }
    status["all_ready"] = all(status.values())
    http_code = 200 if status["all_ready"] else 503
    return jsonify(status), http_code


# ──────────────────────────────────────────────────────────────
# GET /api/swarming/predict-from-iot
# Fetches the latest 24 real IoT readings for a device and
# runs LSTM prediction, returning sensor values + prediction.
# ──────────────────────────────────────────────────────────────
@swarming_live_bp.route("/api/swarming/predict-from-iot", methods=["GET"])
def predict_from_iot():
    """
    Fetch latest real sensor readings from the database and run prediction.

    Query params:
      device_id (required) — e.g. "Hive_01"
      limit     (optional) — number of readings to fetch (default 432)
    """
    device_id = request.args.get("device_id")
    if not device_id:
        return jsonify({"error": "device_id query parameter is required"}), 400

    try:
        limit = int(request.args.get("limit", 432))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400
    # Backfilling a complete 24-hour chart needs 24 hours of model context
    # before the first chart point, so fetch at least 288 ten-minute rows.
    limit = min(max(limit, 288), 5000)

    # ── Load DB settings from env ─────────────────────────────
    SCHEMA = os.getenv("IOT_SCHEMA", "public")
    TABLE = os.getenv("IOT_SENSOR_TABLE", "beehive_readings")
    HIVE_COL = os.getenv("IOT_HIVE_COLUMN", "device_id")
    TIME_COL = os.getenv("IOT_TIMESTAMP_COLUMN", "recorded_at")
    TEMP_COL = os.getenv("IOT_TEMPERATURE_COLUMN", "internal_temp")
    HUM_COL = os.getenv("IOT_HUMIDITY_COLUMN", "internal_humidity")
    CO2_COL = os.getenv("IOT_CO2_COLUMN", "internal_co2")
    WEIGHT_COL = os.getenv("IOT_WEIGHT_COLUMN", "total_weight")
    EXT_T_COL = os.getenv("IOT_EXTERNAL_TEMPERATURE_COLUMN", "external_temp")
    EXT_H_COL = os.getenv("IOT_EXTERNAL_HUMIDITY_COLUMN", "external_humidity")
    BAT_COL = os.getenv("IOT_BATTERY_VOLTAGE_COLUMN", "battery_voltage")

    try:
        # ── Query latest readings ─────────────────────────────
        engine = get_engine()

        sql = text(f"""
            SELECT
                {TIME_COL}   AS recorded_at,
                {TEMP_COL}   AS internal_temp,
                {HUM_COL}    AS internal_humidity,
                {CO2_COL}    AS internal_co2,
                {WEIGHT_COL} AS total_weight,
                {EXT_T_COL}  AS external_temp,
                {EXT_H_COL}  AS external_humidity,
                {BAT_COL}    AS battery_voltage
            FROM {SCHEMA}.{TABLE}
            WHERE {HIVE_COL} = :device
            ORDER BY {TIME_COL} DESC
            LIMIT :lim
        """)

        df = pd.read_sql(sql, engine, params={"device": device_id, "lim": limit})
        df = df.sort_values("recorded_at")  # oldest → newest

        if len(df) < 144:
            return jsonify(
                {
                    "error": f"Not enough data: only {len(df)} readings found (need ≥144 ten-minute readings).",
                    "received": len(df),
                    "required": 144,
                }
            ), 422

        # ── Grab the LATEST single reading for the display panel ──
        latest_row = df.iloc[-1]
        latest_sensor = {
            "internal_temperature_c": round(
                float(latest_row["internal_temp"]) if pd.notna(latest_row["internal_temp"]) else 0,
                2,
            ),
            "internal_humidity_pct": round(
                float(latest_row["internal_humidity"])
                if pd.notna(latest_row["internal_humidity"])
                else 0,
                2,
            ),
            "co2_ppm": round(
                float(latest_row["internal_co2"]) if pd.notna(latest_row["internal_co2"]) else 0, 2
            ),
            "hive_weight_kg": round(
                float(latest_row["total_weight"]) if pd.notna(latest_row["total_weight"]) else 0, 2
            ),
            "external_temperature_c": round(
                float(latest_row["external_temp"]) if pd.notna(latest_row["external_temp"]) else 0,
                2,
            ),
            "external_humidity_pct": round(
                float(latest_row["external_humidity"])
                if pd.notna(latest_row["external_humidity"])
                else 0,
                2,
            ),
            "battery_voltage": round(
                float(latest_row["battery_voltage"])
                if pd.notna(latest_row["battery_voltage"])
                else 0,
                2,
            ),
            "recorded_at": str(latest_row["recorded_at"]),
        }

        # Build the full raw history for the LSTM. pelt_live.py resamples this
        # history to hourly means and selects the latest 24 hourly rows.
        readings_for_lstm = []
        for _, row in df.iterrows():
            readings_for_lstm.append(
                {
                    "recorded_at": str(row["recorded_at"]),
                    "internal_temperature_c": float(row["internal_temp"])
                    if pd.notna(row["internal_temp"])
                    else 35.0,
                    "internal_humidity_pct": float(row["internal_humidity"])
                    if pd.notna(row["internal_humidity"])
                    else 65.0,
                    "co2_ppm": float(row["internal_co2"])
                    if pd.notna(row["internal_co2"])
                    else 1200.0,
                    "hive_weight_kg": float(row["total_weight"])
                    if pd.notna(row["total_weight"])
                    else 32.5,
                    "external_temperature_c": float(row["external_temp"])
                    if pd.notna(row["external_temp"])
                    else 28.0,
                    "external_humidity_pct": float(row["external_humidity"])
                    if pd.notna(row["external_humidity"])
                    else 55.0,
                    "rainfall_mm_hour": 0.0,
                    "wind_speed_mps": 0.0,
                }
            )

        # ── Run LSTM prediction ───────────────────────────────
        predictor = _get_predictor()
        prediction = predictor.predict(device_id, readings_for_lstm)

        # ── ENHANCED: Risk Classification with Softmax ────────
        from .live_prediction.risk_classifier import create_risk_classifier

        # Create risk classifier
        risk_classifier = create_risk_classifier()

        # Preserve the raw LSTM output and add an explicit PELT contribution.
        probability = prediction.get("probability", 0)
        combination = risk_classifier.combine_lstm_and_pelt(
            lstm_probability=probability,
            pelt_snapshot=prediction.get("pelt_snapshot", {}),
        )

        # Classify with Softmax and identify factors
        classification = risk_classifier.classify_with_factors(
            probability=combination["combined_probability"],
            sensor_values=latest_sensor,  # Use latest sensor values for factor analysis
        )

        # Update prediction with enhanced classification
        prediction.update(combination)
        prediction["combined_probability"] = combination["combined_probability"]
        prediction["risk_percentage"] = combination["risk_percentage"]
        prediction["risk_level"] = classification["risk_level"]
        prediction["warning"] = classification.get("message", "")
        prediction["event_window"] = classification.get("event_window", {})
        prediction["key_factors"] = classification.get("key_factors", [])
        prediction["recommendations"] = classification.get("recommendations", [])
        prediction["softmax_probabilities"] = classification.get("softmax_probabilities", {})
        prediction["formula_used"] = combination["formula"]
        prediction["predicted_class"] = (
            "Swarming"
            if combination["combined_probability"] >= prediction.get("threshold_used", 0.70)
            else "No Swarming"
        )

        # Tie the risk point to the latest IoT reading. Repeated manual
        # refreshes for the same reading update one record rather than
        # creating duplicate timeline points.
        prediction["data_timestamp"] = latest_sensor["recorded_at"]

        # Store the completed combined-risk result for the timeline.
        try:
            save_risk_prediction(
                device_id=device_id,
                prediction=prediction,
            )
        except (OSError, TypeError, ValueError):
            logger.exception(
                "Could not save swarming risk history for device %s",
                device_id,
            )

        # Recover prediction points missed while the local backend was closed.
        # The helper skips timestamps that are already stored.
        backfill_summary = {
            "candidates": 0,
            "created": 0,
            "skipped": 0,
            "failed": 0,
        }
        try:
            backfill_summary = backfill_missing_risks(
                device_id=device_id,
                frame=df,
                predictor=predictor,
                risk_classifier=risk_classifier,
            )
            logger.info(
                "Risk backfill for %s: %s",
                device_id,
                backfill_summary,
            )
        except Exception:
            logger.exception(
                "Could not backfill swarming risk history for device %s",
                device_id,
            )

        # Log the classification
        logger.info(
            "Risk classification for %s: %s (combined %.1f%%; LSTM %.1f%%; PELT %.1f%%)",
            device_id,
            classification["risk_level"],
            combination["risk_percentage"],
            combination["lstm_risk_percentage"],
            combination["pelt_risk_percentage"],
        )

        # ── Build 3-day forecast using LSTM + PELT Ensemble ──
        current_risk = prediction["risk_percentage"]

        # ── Helper functions for ensemble forecast ──────────────
        def _history_at_cutoff(readings, hours_back):
            """Return readings available at a historical hourly cutoff."""
            parsed = []
            for item in readings:
                timestamp = pd.to_datetime(item.get("recorded_at"), utc=True, errors="coerce")
                if pd.notna(timestamp):
                    parsed.append((timestamp, item))
            if not parsed:
                return []
            latest_time = max(timestamp for timestamp, _ in parsed)
            cutoff = latest_time - pd.Timedelta(hours=hours_back)
            return [item for timestamp, item in parsed if timestamp <= cutoff]

        def _get_lstm_trend(readings, model):
            """Estimate LSTM risk change in percentage points per hour."""
            from .live_prediction.preprocessing import build_sequence

            probabilities = []
            time_axis = []
            for hours_back in (18, 12, 6, 0):
                history = _history_at_cutoff(readings, hours_back)
                try:
                    seq = build_sequence(history)
                    probabilities.append(float(model.predict(seq, verbose=0)[0][0]) * 100)
                    time_axis.append(-hours_back)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Historical LSTM prediction failed: %s", exc)
            if len(probabilities) >= 3:
                slope = float(np.polyfit(np.array(time_axis), np.array(probabilities), 1)[0])
                return {"current": probabilities[-1], "trend": slope}
            return {"current": current_risk, "trend": 0}

        def _get_pelt_trend(readings):
            """Get PELT feature trends."""
            from .live_prediction.pelt_live import generate_pelt_features

            densities, days_since, time_axis = [], [], []
            for hours_back in (18, 12, 6, 0):
                history = _history_at_cutoff(readings, hours_back)
                try:
                    pelt_df = generate_pelt_features(history)
                    last = pelt_df.iloc[-1]
                    densities.append(float(last["breakpoint_density"]))
                    days_since.append(float(last["days_since_breakpoint"]))
                    time_axis.append(-hours_back)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Historical PELT calculation failed: %s", exc)
            if len(densities) >= 3:
                x = np.array(time_axis)
                return {
                    "density": densities[-1],
                    "days_since": days_since[-1],
                    "density_trend": float(np.polyfit(x, np.array(densities), 1)[0]),
                    "days_trend": float(np.polyfit(x, np.array(days_since), 1)[0]),
                }
            return {"density": 0, "days_since": 0, "density_trend": 0, "days_trend": 0}

        def _get_sensor_trend(readings):
            """Get trend from sensor data."""
            sensor_df = pd.DataFrame(readings)
            if "recorded_at" not in sensor_df or "internal_temperature_c" not in sensor_df:
                return 0
            sensor_df["recorded_at"] = pd.to_datetime(
                sensor_df["recorded_at"], utc=True, errors="coerce"
            )
            sensor_df["internal_temperature_c"] = pd.to_numeric(
                sensor_df["internal_temperature_c"], errors="coerce"
            )
            hourly_temp = (
                sensor_df.dropna(subset=["recorded_at"])
                .set_index("recorded_at")["internal_temperature_c"]
                .resample("1h")
                .mean()
                .interpolate()
                .dropna()
                .tail(24)
            )
            if len(hourly_temp) >= 3:
                return float(np.polyfit(np.arange(len(hourly_temp)), hourly_temp.to_numpy(), 1)[0])
            return 0

        def _get_daily_pattern(hours_ahead):
            """Get daily activity pattern."""
            current_hour = datetime.now(UTC).hour
            forecast_hour = (current_hour + hours_ahead) % 24
            daily_cycle = np.sin((forecast_hour - 6) * np.pi / 12)
            return 7.5 + (daily_cycle * 7.5)  # Range: 0-15%

        # ── Generate Ensemble Forecast ──
        try:
            # Get LSTM model
            model = _get_predictor()._model

            # Get trends from all methods
            lstm_trend = _get_lstm_trend(readings_for_lstm, model)
            pelt_trend = _get_pelt_trend(readings_for_lstm)
            sensor_trend = _get_sensor_trend(readings_for_lstm)

            forecast_days = []
            for day in range(1, 4):
                hours_ahead = day * 24

                # 1. LSTM Projection (40% weight)
                lstm_projection = lstm_trend["current"] + (
                    lstm_trend["trend"] * hours_ahead * np.exp(-hours_ahead / 72)
                )
                lstm_projection = max(0, min(100, lstm_projection))

                # 2. PELT Projection (30% weight)
                future_density = pelt_trend["density"] + (
                    pelt_trend["density_trend"] * hours_ahead / 24
                )
                density_risk = min(100, future_density * 15)
                days_risk = max(0, 40 - pelt_trend["days_since"] * 1.5)
                pelt_projection = density_risk * 0.6 + days_risk * 0.4
                pelt_projection = max(0, min(100, pelt_projection))

                # 3. Trend Projection (20% weight) - uses current_risk as baseline
                risk_change = sensor_trend * (hours_ahead / 48) * 10
                trend_projection = max(0, min(100, current_risk + risk_change))

                # 4. Daily Pattern (10% weight)
                daily_pattern = _get_daily_pattern(hours_ahead)

                # Ensemble: Weighted Average
                ensemble_risk = (
                    0.40 * lstm_projection
                    + 0.30 * pelt_projection
                    + 0.20 * trend_projection
                    + 0.10 * daily_pattern
                )

                final_risk = max(0, min(100, ensemble_risk))

                # Use RiskClassifier for risk level and color
                forecast_classification = risk_classifier.classify_from_probability(
                    final_risk / 100
                )

                forecast_days.append(
                    {
                        "day": day,
                        "risk": round(final_risk, 1),
                        "level": forecast_classification["risk_level"],
                        "color": risk_classifier.get_risk_color(
                            forecast_classification["risk_level"]
                        ),
                    }
                )

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Ensemble forecast failed: {e}, using fallback")
            # Fallback: simple projection with daily patterns
            forecast_days = []
            for day in range(1, 4):
                daily_cycle = 8 * np.sin((day * 24 - 12) * np.pi / 24)
                risk = max(0, min(100, current_risk + day * 2 + daily_cycle))
                forecast_classification = risk_classifier.classify_from_probability(risk / 100)
                forecast_days.append(
                    {
                        "day": day,
                        "risk": round(risk, 1),
                        "level": forecast_classification["risk_level"],
                        "color": risk_classifier.get_risk_color(
                            forecast_classification["risk_level"]
                        ),
                    }
                )

        # Build historical sensor readings for the live trend chart.
        # The frontend expects these sensor fields in every item returned
        # through data.readings. Previously, only recorded_at was returned,
        # so RealtimeSensorTrendsTimeline rejected every chart point.
        readings_out = []
        for _, row in df.iterrows():
            readings_out.append(
                {
                    "recorded_at": str(row["recorded_at"]),
                    "internal_temperature_c": round(float(row["internal_temp"]), 2)
                    if pd.notna(row["internal_temp"])
                    else 35.0,
                    "internal_humidity_pct": round(float(row["internal_humidity"]), 2)
                    if pd.notna(row["internal_humidity"])
                    else 65.0,
                    "co2_ppm": round(float(row["internal_co2"]), 2)
                    if pd.notna(row["internal_co2"])
                    else 1200.0,
                    "hive_weight_kg": round(float(row["total_weight"]), 2)
                    if pd.notna(row["total_weight"])
                    else 32.5,
                    "external_temperature_c": round(float(row["external_temp"]), 2)
                    if pd.notna(row["external_temp"])
                    else 28.0,
                    "external_humidity_pct": round(float(row["external_humidity"]), 2)
                    if pd.notna(row["external_humidity"])
                    else 55.0,
                    "battery_voltage": round(float(row["battery_voltage"]), 2)
                    if pd.notna(row["battery_voltage"])
                    else 0.0,
                }
            )

        # ── Build response with enhanced prediction ──────────
        response = {
            "device_id": device_id,
            "latest_sensor": latest_sensor,
            "prediction": prediction,
            "forecast": forecast_days,
            "readings_used": len(readings_for_lstm),
            "readings": readings_out,
            "event_window": prediction.get("event_window", {}),
            "key_factors": prediction.get("key_factors", []),
            "recommendations": prediction.get("recommendations", []),
            "softmax_probabilities": prediction.get("softmax_probabilities", {}),
            "formula_used": prediction.get("formula_used", ""),
            "backfill": backfill_summary,
        }

        return jsonify(response), 200

    except Exception as e:
        logger.exception("Error in predict_from_iot")
        return jsonify(
            {
                "error": "Failed to fetch IoT data or run prediction.",
                "details": str(e),
            }
        ), 500


# ──────────────────────────────────────────────────────────────
# GET /api/swarming/risk-history
# Returns stored swarming risks from the local JSON history.
# ──────────────────────────────────────────────────────────────
@swarming_live_bp.get("/api/swarming/risk-history")
def swarming_risk_history():
    """Return stored swarming risks for the selected device."""
    device_id = request.args.get("device_id", "").strip()

    if not device_id:
        return jsonify(
            {
                "error": "device_id query parameter is required",
                "history": [],
            }
        ), 400

    try:
        minutes = int(request.args.get("minutes", 1440))
        limit = int(request.args.get("limit", 5000))
    except ValueError:
        return jsonify(
            {
                "error": "minutes and limit must be integers",
                "history": [],
            }
        ), 400

    history = get_risk_history(
        device_id=device_id,
        minutes=minutes,
        limit=limit,
    )

    return jsonify(
        {
            "device_id": device_id,
            "minutes": min(max(minutes, 1), 1440),
            "count": len(history),
            "history": history,
        }
    ), 200


@swarming_live_bp.get("/api/eda-swarming")
def swarming_eda_dashboard():
    path = _EDA_DIR / "dashboard.json"
    if not path.exists():
        return jsonify({"error": "Swarming EDA dashboard not found."}), 404
    with path.open("r", encoding="utf-8") as handle:
        return jsonify(json.load(handle))


@swarming_live_bp.get("/api/eda-swarming/images")
def swarming_eda_images():
    images = _EDA_DIR / "images"
    return jsonify(sorted(path.name for path in images.glob("*.png")))


@swarming_live_bp.get("/api/eda-swarming/images/<path:filename>")
def swarming_eda_image(filename):
    return send_from_directory(_EDA_DIR / "images", filename)


@swarming_live_bp.get("/api/eda-swarming/report")
def swarming_eda_report():
    path = _EDA_DIR / "reports" / "feature_analysis_summary.txt"
    if not path.exists():
        return jsonify({"error": "Swarming EDA report not found."}), 404
    return jsonify({"report": path.read_text(encoding="utf-8")})
