#!/usr/bin/env python3
from config import settings
import joblib
import pandas as pd
import sys
from pathlib import Path

# Ensure src on path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Paths
model_path = "models/xgb_model.joblib"
parquet_path = "data/processed/full_data.parquet"

# Load model
model = joblib.load(model_path)
print(f"Loaded model from {model_path}")

# Load sample data
df = pd.read_parquet(parquet_path)
sample = df.head(5)
print("Sample data:")
print(sample)

# Run prediction
preds = model.predict(sample)
print("Predictions:", preds)
