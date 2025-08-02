# tests/conftest.py
import pytest
import joblib
import pandas as pd
from xgboost import XGBClassifier
from src.config import settings


@pytest.fixture(autouse=True)
def dummy_model(tmp_path_factory, monkeypatch):
    """
    Create a minimal XGB model and point settings.model_output_path at it before any tests import src.api.predict.
    """
    tmpdir = tmp_path_factory.mktemp("model")
    model_file = tmpdir / "dummy_model.joblib"
    # Train trivial model on two rows matching feature_columns
    data = {col: [0, 1] for col in settings.feature_columns}
    df = pd.DataFrame(data)
    y = pd.Series([0, 1])
    model = XGBClassifier(use_label_encoder=False,
                          eval_metric='mlogloss', n_estimators=1, max_depth=1)
    model.fit(df, y)
    joblib.dump(model, model_file)
    monkeypatch.setattr(settings, "model_output_path", str(model_file))
    return model_file
