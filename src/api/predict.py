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
    df = pd.DataFrame([item])[settings.feature_columns]
    pred = get_model().predict(df)[0]
    return {"prediction": int(pred)}


def predict_batch(items: list[dict]) -> list:
    df = pd.DataFrame(items)[settings.feature_columns]
    preds = get_model().predict(df)
    return [{"prediction": int(p)} for p in preds]
