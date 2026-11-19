"""Scoring script for the credit card fraud detection endpoint.

Azure Machine Learning calls init() once when the container starts and
run() for every scoring request.
"""
import json
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger("fraud-endpoint")
logger.setLevel(logging.INFO)

model = scaler = metadata = None


def _find(filename):
    """Locate a file inside the registered model folder."""
    for root, _, files in os.walk(os.environ["AZUREML_MODEL_DIR"]):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(filename)


def init():
    global model, scaler, metadata
    model = xgb.XGBClassifier()
    model.load_model(_find("fraud_xgb_model.json"))
    scaler = joblib.load(_find("amount_scaler.joblib"))
    with open(_find("model_metadata.json")) as f:
        metadata = json.load(f)
    logger.info("Model loaded: %s, threshold %.4f",
                metadata["model_name"], metadata["decision_threshold"])


def _preprocess(df):
    """Apply exactly the same feature engineering as in training."""
    hour = (df["Time"] // 3600) % 24
    out = df.copy()
    out["log_Amount"] = scaler.transform(
        pd.DataFrame({"log_Amount": np.log1p(df["Amount"])}))[:, 0]
    out["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out[metadata["model_features"]]


def run(raw_data):
    start = time.perf_counter()
    try:
        records = json.loads(raw_data)["data"]
        df = pd.DataFrame(records)
        missing = [c for c in metadata["raw_input_columns"] if c not in df.columns]
        if missing:
            return {"error": f"Missing required fields: {missing}"}
        if df[metadata["raw_input_columns"]].isna().any().any():
            return {"error": "Input contains missing values."}

        proba = model.predict_proba(_preprocess(df))[:, 1]
        threshold = metadata["decision_threshold"]
        results = [{"fraud_probability": round(float(p), 6),
                    "is_fraud": bool(p >= threshold)} for p in proba]

        # Operational log only: counts and latency, never transaction data
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Scored %d transactions, %d flagged, %.2f ms",
                    len(results), sum(r["is_fraud"] for r in results), latency_ms)
        return {"model": metadata["model_name"], "threshold": threshold,
                "predictions": results}
    except Exception as exc:
        logger.error("Scoring failed: %s", exc)
        return {"error": str(exc)}
