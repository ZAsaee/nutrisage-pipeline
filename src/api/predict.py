# src/api/predict.py
import joblib
import pandas as pd
from src.config import settings


_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(settings.model_output_path)
    return _model


def predict_single(item: dict) -> dict:
    # Convert single record to DataFrame
    df = pd.DataFrame([item])
    model = get_model()
    # Enforce correct feature columns and order
    if hasattr(model, 'feature_names_in_'):
        df = df[model.feature_names_in_]
    # Predict and return one result
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}


def predict_batch(items: list[dict]) -> list:
    # Convert list of records to DataFrame
    df = pd.DataFrame(items)
    model = get_model()
    # Enforce correct feature columns and order
    if hasattr(model, 'feature_names_in_'):
        df = df[model.feature_names_in_]
    # Predict in batch
    preds = model.predict(df)
    return [{"prediction": int(p)} for p in preds]
