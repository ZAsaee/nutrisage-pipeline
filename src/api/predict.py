# src/api/predict.py
import joblib
import pandas as pd
from src.config import settings


_model = joblib.load(settings.model_output_path)


def predict_single(item: dict) -> dict:
    df = pd.DataFrame([item])[settings.feature_columns]
    pred = _model.predict(df)[0]
    return {"prediction": int(pred)}


def predict_batch(items: list[dict]) -> list:
    df = pd.DataFrame(items)[settings.feature_columns]
    preds = _model.predict(df)
    return [{"prediction": int(p)} for p in preds]
