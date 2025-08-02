# tests/test_modeling.py
import pandas as pd
from src.modeling.training import train_model, bayesian_search
from xgboost import XGBClassifier
from src.config import settings


def make_df():
    data = {col: [0, 1, 0, 1] for col in settings.feature_columns}
    data[settings.label_column] = [0, 1, 0, 1]
    return pd.DataFrame(data)


def test_train_model_returns_model():
    df = make_df()
    X, y = df[settings.feature_columns], df[settings.label_column]
    model = train_model(X, y, n_estimators=5, max_depth=2, random_state=0)
    assert isinstance(model, XGBClassifier)
    preds = model.predict(X)
    assert all(preds == y)


def test_bayesian_search_minimal():
    # Run bayes search with n_iter=1 for speed
    df = make_df()
    X, y = df[settings.feature_columns], df[settings.label_column]
    model = bayesian_search(X, y)
    assert hasattr(model, 'predict')
