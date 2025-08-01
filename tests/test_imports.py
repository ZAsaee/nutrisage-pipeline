# tests/test_imports.py
import pytest


def test_imports():
    # Smoke test: ensure modules are importable
    import src.config
    import src.data.storage
    import src.data.utils
    import src.data.dataset
    import src.preprocessing.steps
    import src.preprocessing.pipeline
    import src.modeling.training
    import src.modeling.feature_importance
    import src.api.app
    import src.api.predict
