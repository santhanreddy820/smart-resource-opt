# src/api.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ================================
# FastAPI App
# ================================
app = FastAPI(title="Smart Resource Optimization API")

# ================================
# Models Loading
# ================================
def load_model_safe(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"ERROR loading model {path}: {e}")
    return None

clf = load_model_safe("models/fault_rf.pkl")        # classifier
forecast_model = load_model_safe("models/forecast_h1.pkl")  # forecast model (if needed)

# ================================
# Request Models
# ================================
class Reading(BaseModel):
    ts: str
    device_id: str
    power_kW: float

class BatchRequest(BaseModel):
    readings: List[Reading]

# ================================
# Root & Health Check
# ================================
@app.get("/")
def root():
    return {"message": "API running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "classifier_loaded": clf is not None,
        "forecast_loaded": forecast_model is not None
    }

# ============================================================
# STRICT FEATURE BUILDER — matches your model EXACTLY
# Your model expects ONLY these 5 features:
# ['power_kW', 'power_lag_1', 'power_roll_1h', 'hour', 'dow']
# ============================================================

EXPECTED_FEATURES = ['power_kW', 'power_lag_1', 'power_roll_1h', 'hour', 'dow']

def create_features_from_reading(r: Reading):
    """Convert one reading into the exact 5 features classifier expects."""
    ts = pd.to_datetime(r.ts)

    # Since this is real-time (no history), generate minimal defaults.
    return {
        "power_kW": r.power_kW,
        "power_lag_1": r.power_kW,        # fallback lag
        "power_roll_1h": r.power_kW,      # fallback rolling average
        "hour": ts.hour,
        "dow": ts.dayofweek,
    }

def make_feature_df(readings: List[Reading]):
    rows = [create_features_from_reading(r) for r in readings]
    df = pd.DataFrame(rows)

    # enforce exact model feature order
    df = df[EXPECTED_FEATURES]

    return df

# ================================
# Single Prediction
# ================================
@app.post("/predict")
def predict_one(r: Reading):
    if clf is None:
        return {"error": "Classifier model not loaded"}

    df = make_feature_df([r])
    prob = clf.predict_proba(df)[0][1]

    return {
        "device_id": r.device_id,
        "ts": r.ts,
        "fault_probability": float(prob),
        "recommended_action": "Inspect" if prob > 0.7 else "Monitor"
    }

# ================================
# Batch Prediction
# ================================
@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    if clf is None:
        return {"error": "Classifier model not loaded"}

    df = make_feature_df(req.readings)
    preds = clf.predict_proba(df)[:, 1]

    results = []
    for r, p in zip(req.readings, preds):
        results.append({
            "device_id": r.device_id,
            "ts": r.ts,
            "fault_probability": float(p),
            "recommended_action": "Inspect" if p > 0.7 else "Monitor"
        })

    return {"results": results}
