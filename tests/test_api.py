# tests/test_api.py
from fastapi.testclient import TestClient
from src.api.app import app
from src.config import settings

client = TestClient(app)


def test_predict_single():
    payload = {col: 1.0 for col in settings.feature_columns}
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    assert 'prediction' in response.json()
