# tests/test_feature_importance_unit.py
import pandas as pd
from src.config import settings
import joblib
from xgboost import XGBClassifier
import sys
import os

from src.modeling.feature_importance import main as fi_main


def test_feature_importance_function(tmp_path, monkeypatch):
    # Train a simple model
    df = pd.DataFrame({col: [0, 1, 0, 1] for col in settings.feature_columns})
    df[settings.label_column] = [0, 1, 0, 1]
    X, y = df[settings.feature_columns], df[settings.label_column]
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)

    # Dump model to temp file
    model_file = tmp_path / 'model.joblib'
    joblib.dump(model, model_file)

    # Prepare preprocessed data file
    data_file = tmp_path / 'data.parquet'
    df.to_parquet(data_file)

    # Output path
    output_file = tmp_path / 'fi.csv'

    # Monkey-patch settings (if fi_main uses settings.model_output_path)
    monkeypatch.setattr(settings, 'model_output_path', str(model_file))

    # Build argv for argparse-based CLI
    sys_argv = sys.argv.copy()
    sys.argv = [sys_argv[0],
                '--model-input', str(model_file),
                '--data-input', str(data_file),
                '--output', str(output_file)]
    try:
        fi_main()
    finally:
        sys.argv = sys_argv

    # Validate output CSV
    assert output_file.exists()
    fi_df = pd.read_csv(output_file)
    assert set(fi_df.columns) == {'feature', 'importance'}
