"""
Minimal tests for preprocessing pipeline using sample data.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.preprocessing import NutritionDataPreprocessor


def test_preprocessing_with_sample_data():
    """Test preprocessing pipeline with sample data."""
    sample_path = Path(__file__).parent / "data" / "sample_nutrition_data.csv"
    preprocessor = NutritionDataPreprocessor()
    
    features_df, target_df, feature_columns, target_col = preprocessor.preprocess(
        sample_path, save_intermediate=False
    )
    
    assert isinstance(features_df, pd.DataFrame)
    assert isinstance(target_df, pd.DataFrame)
    assert len(feature_columns) > 0
    assert target_col == 'nutrition_grade_fr'
    assert len(features_df) == len(target_df)


def test_preprocessing_output_quality():
    """Test quality of preprocessing output."""
    sample_path = Path(__file__).parent / "data" / "sample_nutrition_data.csv"
    preprocessor = NutritionDataPreprocessor()
    
    features_df, target_df, feature_columns, target_col = preprocessor.preprocess(
        sample_path, save_intermediate=False
    )
    
    # Check features
    assert all(col in features_df.columns for col in feature_columns)
    assert not features_df.empty
    
    # Check target
    assert target_df[target_col].nunique() >= 2
    assert target_df[target_col].notna().all()


def test_preprocessing_no_missing_values():
    """Test that preprocessing handles missing values."""
    sample_path = Path(__file__).parent / "data" / "sample_nutrition_data.csv"
    preprocessor = NutritionDataPreprocessor()
    
    features_df, target_df, feature_columns, target_col = preprocessor.preprocess(
        sample_path, save_intermediate=False
    )
    
    # Check no missing values in features
    assert not features_df[feature_columns].isnull().any().any()
    assert not target_df[target_col].isnull().any() 