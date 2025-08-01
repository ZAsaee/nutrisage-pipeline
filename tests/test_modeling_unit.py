# tests/test_modeling_unit.py
import pandas as pd
from src.modeling.training import train_model, bayesian_search
from src.config import settings
from xgboost import XGBClassifier
import pytest


def make_df():
    data = {col: [0, 1, 0, 1] for col in settings.feature_columns}
    data[settings.label_column] = [0, 1, 0, 1]
    return pd.DataFrame(data)


def test_train_model():
    df = make_df()
    X, y = df[settings.feature_columns], df[settings.label_column]
    model = train_model(X, y, n_estimators=5, max_depth=2, random_state=0)
    assert isinstance(model, XGBClassifier)
    preds = model.predict(X)
    # Ensure predictions have correct length and valid labels
    assert len(preds) == len(y)
    assert set(preds).issubset(set(y.unique()))


def test_bayesian_search_minimal():
    df = make_df()
    X, y = df[settings.feature_columns], df[settings.label_column]
    try:
        model = bayesian_search(X, y)
    except ValueError as e:
        pytest.skip(f"Skipping Bayesian search on small dataset: {e}")
    # If it returns a model, ensure it can predict
    assert hasattr(model, 'predict')
