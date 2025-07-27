"""
Minimal tests for training pipeline using sample data.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock

from src.preprocessing import NutritionDataPreprocessor
from src.modeling.train import train_xgboost_model, save_model_and_metadata


def test_model_training_with_sample_data():
    """Test model training with sample data."""
    sample_path = Path(__file__).parent / "data" / "sample_nutrition_data.csv"
    
    # Preprocess data
    preprocessor = NutritionDataPreprocessor()
    features_df, target_df, feature_columns, target_col = preprocessor.preprocess(
        sample_path, save_intermediate=False
    )
    
    # Train model
    X = features_df[feature_columns]
    y = target_df[target_col]
    model, X_test, y_test, y_pred, feature_importance = train_xgboost_model(
        X, y, feature_columns, tune_hyperparameters=False
    )
    
    # Check model
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    
            # Check predictions
        assert len(y_pred) == len(X_test)
        assert all(pred in ['a', 'b', 'c', 'd', 'e'] for pred in y_pred)


def test_model_saving_and_loading():
    """Test model saving and loading."""
    import tempfile
    from src.modeling.predict import load_model_and_metadata
    
    sample_path = Path(__file__).parent / "data" / "sample_nutrition_data.csv"
    
    # Train model
    preprocessor = NutritionDataPreprocessor()
    features_df, target_df, feature_columns, target_col = preprocessor.preprocess(
        sample_path, save_intermediate=False
    )
    
    X = features_df[feature_columns]
    y = target_df[target_col]
    model, X_test, y_test, y_pred, feature_importance = train_xgboost_model(
        X, y, feature_columns, tune_hyperparameters=False
    )
    
    # Save model
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.pkl"
        metadata_path = Path(temp_dir) / "test_metadata.pkl"
        
        save_model_and_metadata(
            model, feature_columns, Mock(), feature_importance,
            model_path, metadata_path
        )
        
        # Load model
        loaded_model, loaded_metadata = load_model_and_metadata(model_path, metadata_path)
        
        # Check loaded model
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
        assert 'feature_columns' in loaded_metadata


def test_feature_importance():
    """Test feature importance calculation."""
    sample_path = Path(__file__).parent / "data" / "sample_nutrition_data.csv"
    
    # Train model
    preprocessor = NutritionDataPreprocessor()
    features_df, target_df, feature_columns, target_col = preprocessor.preprocess(
        sample_path, save_intermediate=False
    )
    
    X = features_df[feature_columns]
    y = target_df[target_col]
    model, X_test, y_test, y_pred, feature_importance = train_xgboost_model(
        X, y, feature_columns, tune_hyperparameters=False
    )
    
    # Check feature importance
    assert isinstance(feature_importance, pd.DataFrame)
    assert len(feature_importance) > 0
    assert 'feature' in feature_importance.columns
    assert 'importance' in feature_importance.columns 