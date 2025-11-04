#!/usr/bin/env python3
"""
app.py

Flask API for UDP DDoS detection.

- Loads a sklearn Pipeline (StandardScaler + RandomForest) and feature list
- Source: local artifacts/ or S3, controlled by environment variables
- Endpoints:
    GET  /           -> health check
    GET  /features   -> global feature importances
    POST /predict    -> JSON {feature: value, ...} -> {attack, attack_probability, top_features}

Env vars (for AWS / prod):
    S3_BUCKET        : if set, load model/feats from this bucket
    S3_MODEL_KEY     : S3 key for model (default: udp_ddos_pipeline.pkl)
    S3_FEAT_KEY      : S3 key for features (default: udp_features.json)

Local dev:
    Place artifacts/udp_ddos_pipeline.pkl and artifacts/udp_features.json
    Run: uv run app.py
"""

import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import boto3
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Config
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_MODEL_KEY = os.environ.get("S3_MODEL_KEY", "udp_ddos_pipeline.pkl")
S3_FEAT_KEY = os.environ.get("S3_FEAT_KEY", "udp_features.json")

LOCAL_MODEL = Path("artifacts/udp_ddos_pipeline.pkl")
LOCAL_FEAT = Path("artifacts/udp_features.json")

app = Flask(__name__)


def load_from_local():
    if not LOCAL_MODEL.exists() or not LOCAL_FEAT.exists():
        raise SystemExit("Local artifacts not found in artifacts/; run train.py first.")
    log.info("Loading model and features from local artifacts/")
    model = joblib.load(LOCAL_MODEL)
    features = json.load(open(LOCAL_FEAT))
    return model, features


def load_from_s3():
    if not S3_BUCKET:
        raise SystemExit("S3_BUCKET env var not set")
    log.info("Loading model and features from S3 bucket %s", S3_BUCKET)
    s3 = boto3.client("s3")

    # model
    model_obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY)
    model_bytes = model_obj["Body"].read()
    model = joblib.load(io.BytesIO(model_bytes))

    # features
    feat_obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_FEAT_KEY)
    feat_bytes = feat_obj["Body"].read()
    features = json.loads(feat_bytes.decode("utf-8"))

    return model, features


# Load artifacts once at startup
if S3_BUCKET:
    MODEL, FEATURES = load_from_s3()
else:
    MODEL, FEATURES = load_from_local()

# Extract underlying RandomForest and scaler for XAI
try:
    RF = MODEL.named_steps["rf"]
    SCALER = MODEL.named_steps["scaler"]
except Exception as e:  # fallback if someone changed pipeline names
    log.warning("Could not access named steps rf/scaler directly: %s", e)
    RF = None
    SCALER = None


def compute_global_importance() -> List[Dict[str, Any]]:
    """Return sorted global feature importance from RandomForest."""
    if RF is None:
        return []
    importances = RF.feature_importances_
    pairs = list(zip(FEATURES, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": float(imp)} for f, imp in pairs]


GLOBAL_IMPORTANCE = compute_global_importance()


@app.route("/", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "n_features": len(FEATURES),
            "using_s3": bool(S3_BUCKET),
        }
    )


@app.route("/features", methods=["GET"])
def features():
    """Global feature importance (XAI)."""
    return jsonify(GLOBAL_IMPORTANCE)


def compute_local_explanation(row: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Simple per-sample explanation:

    - Build a 1xN DataFrame from the input row
    - Transform with the same scaler used by the model
    - Multiply scaled values by feature_importances_ to get crude scores
    - Return top 5 features by score
    """
    if RF is None or SCALER is None:
        return []

    # Build DF with correct order
    vec = np.array([[row.get(f, 0.0) for f in FEATURES]], dtype=float)
    vec_scaled = SCALER.transform(vec)[0]  # shape (n_features,)

    importances = RF.feature_importances_
    scores = np.abs(vec_scaled * importances)
    # Normalize scores for readability (optional)
    total = scores.sum() or 1.0
    norm_scores = scores / total

    pairs = list(zip(FEATURES, norm_scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:5]
    return [{"feature": f, "relative_score": float(s)} for f, s in top]


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    JSON body: { "feature_name": value, ... }

    Returns:
    {
      "attack": bool,
      "attack_probability": float,
      "top_features": [
          {"feature": ..., "relative_score": ...},
          ...
      ]
    }
    """
    payload = request.get_json(force=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body must be an object"}), 400

    # Build input row in correct order; missing features default to 0.0
    row = {f: float(payload.get(f, 0.0)) for f in FEATURES}
    df = pd.DataFrame([row], columns=FEATURES)

    y_pred = int(MODEL.predict(df)[0])
    proba = float(MODEL.predict_proba(df)[0][1])

    top_feats = compute_local_explanation(row)

    return jsonify(
        {
            "attack": bool(y_pred),
            "attack_probability": proba,
            "top_features": top_feats,
        }
    )


if __name__ == "__main__":
    # Local dev: uv run app.py
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
