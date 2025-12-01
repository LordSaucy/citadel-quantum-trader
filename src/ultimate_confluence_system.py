import json
from typing import Dict

from flask import Blueprint, request, jsonify
from prometheus_client import Gauge

from .config import logger

# ----------------------------------------------------------------------
# Tunable weights – defaults; overridden at runtime via the API
# ----------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "weight_mtf_structure": 0.20,
    "weight_aoi": 0.20,
    "weight_candlestick": 0.10,
    "weight_smc": 0.20,
    "weight_head_shoulders": 0.20,
    "weight_traditional_ta": 0.10,
}

# Prometheus gauge that mirrors the dict (dynamic)
confluence_gauge = Gauge(
    "confluence_parameter",
    "Runtime‑adjustable weight for each CQT component",
    ["parameter"],
)

# Initialise gauges with defaults
for k, v in DEFAULT_WEIGHTS.items():
    confluence_gauge.labels(parameter=k).set(v)

# ----------------------------------------------------------------------
# Flask blueprint – mounted under the main app at /config
# ----------------------------------------------------------------------
bp = Blueprint("confluence", __name__, url_prefix="/config")


@bp.route("/", methods=["GET"])
def dump_all():
    """Return the full weight dict as JSON."""
    return jsonify(DEFAULT_WEIGHTS)


@bp.route("/<key>", methods=["GET"])
def get_one(key: str):
    """Read a single weight."""
    if key not in DEFAULT_WEIGHTS:
        return jsonify({"error": f"unknown key {key}"}), 404
    return jsonify({key: DEFAULT_WEIGHTS[key]})


@bp.route("/<key>", methods=["POST", "PUT", "PATCH"])
def set_one(key: str):
    """Update a single weight – expects JSON { "value": <float> }."""
    if key not in DEFAULT_WEIGHTS:
        return jsonify({"error": f"unknown key {key}"}), 404

    try:
        payload = request.get_json(force=True)
        new_val = float(payload["value"])
    except Exception as exc:  # pragma: no cover
        logger.error("Bad payload for %s: %s", key, exc)
        return jsonify({"error": "invalid payload"}), 400

    # Clamp to 0‑1 range (weights are percentages)
    new_val = max(0.0, min(1.0, new_val))
    DEFAULT_WEIGHTS[key] = new_val
    confluence_gauge.labels(parameter=key).set(new_val)
    logger.info("Weight %s updated to %.3f", key, new_val)
    return jsonify({key: new_val})
