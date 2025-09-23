from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import json


app = FastAPI(title='Heart Disease Prediction API')


# Load model and feature order on startup
MODEL_PATH = 'best_model.pkl'
FEATURE_ORDER_PATH = 'feature_order.json'


model = joblib.load(MODEL_PATH)
with open(FEATURE_ORDER_PATH) as f:
    FEATURE_ORDER = json.load(f)


class Features(BaseModel):
    # Accept a dict of features; we'll validate presence in code
    data: dict


@app.post('/predict')
def predict(features: Features):
    x = features.data
    # Check all required features present
    missing = [f for f in FEATURE_ORDER if f not in x]
    if missing:
        raise HTTPException(status_code=400, detail={'missing_features': missing})

    # Create ordered input
    try:
        arr = np.array([x[f] for f in FEATURE_ORDER], dtype=float).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Prediction
    try:
        proba = model.predict_proba(arr)[0, 1] if hasattr(model, 'predict_proba') else None
        pred = int(model.predict(arr)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {'prediction': pred, 'probability': float(proba) if proba is not None else None}


@app.get('/')
def root():
    return {'message': 'Heart Disease Prediction API is running'}
