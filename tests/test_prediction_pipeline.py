"""
Minimal tests for prediction pipeline using sample data.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
from unittest.mock import Mock

from src.preprocessing import NutritionDataPreprocessor
from src.modeling.train import train_xgboost_model, save_model_and_metadata
from src.modeling.predict import load_model_and_metadata, preprocess_input_data, make_predictions


def test_prediction_with_sample_data():
    """Test prediction pipeline with sample data."""
    sample_path = Path(__file__).parent / "data" / "sample_nutrition_data.csv"
    
    # Train and save model
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
        
        # Load model for prediction
        loaded_model, loaded_metadata = load_model_and_metadata(model_path, metadata_path)
        
        # Create test data
        test_data = pd.DataFrame({
            'fat_100g': [15.0, 5.0],
            'carbohydrates_100g': [20.0, 30.0],
            'proteins_100g': [8.0, 12.0],
            'salt_100g': [0.5, 1.0],
            'sugars_100g': [10.0, 15.0],
            'fiber_100g': [2.0, 3.0],
            'saturated-fat_100g': [5.0, 2.0]
        })
        
        # Make predictions
        X_test_processed = preprocess_input_data(test_data, loaded_metadata, None)
        results = make_predictions(loaded_model, X_test_processed, loaded_metadata)
        
        # Check results
        assert 'predictions' in results
        assert 'predicted_labels' in results
        assert results['num_samples'] == 2
        assert all(pred in ['a', 'b', 'c', 'd', 'e'] for pred in results['predicted_labels'])


def test_single_prediction():
    """Test single sample prediction."""
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
    
    # Create single test sample
    single_data = pd.DataFrame({
        'fat_100g': [12.0],
        'carbohydrates_100g': [25.0],
        'proteins_100g': [10.0],
        'salt_100g': [0.8],
        'sugars_100g': [12.0],
        'fiber_100g': [2.5],
        'saturated-fat_100g': [4.0]
    })
    
    # Make prediction
    metadata = {'feature_columns': feature_columns}
    X_single = preprocess_input_data(single_data, metadata, None)
    results = make_predictions(model, X_single, metadata)
    
    # Check single prediction
    assert results['num_samples'] == 1
    assert len(results['predicted_labels']) == 1
    assert results['predicted_labels'][0] in ['a', 'b', 'c', 'd', 'e']


def test_prediction_probabilities():
    """Test prediction with probabilities."""
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
    
    # Create test data
    test_data = pd.DataFrame({
        'fat_100g': [15.0],
        'carbohydrates_100g': [20.0],
        'proteins_100g': [8.0],
        'salt_100g': [0.5],
        'sugars_100g': [10.0],
        'fiber_100g': [2.0],
        'saturated-fat_100g': [5.0]
    })
    
    # Make prediction with probabilities
    metadata = {'feature_columns': feature_columns}
    X_test_processed = preprocess_input_data(test_data, metadata, None)
    results = make_predictions(model, X_test_processed, metadata, return_probabilities=True)
    
    # Check probabilities
    assert 'probabilities' in results
    assert results['probabilities'] is not None
    assert len(results['probabilities']) == 1 