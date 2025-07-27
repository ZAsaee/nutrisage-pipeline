import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.preprocessing import NutritionDataPreprocessor


class TestNutritionDataPreprocessor:
    """Test cases for the NutritionDataPreprocessor class."""
    
    def test_default_config(self):
        """Test that default configuration is loaded correctly."""
        preprocessor = NutritionDataPreprocessor()
        
        assert 'columns_to_drop' in preprocessor.config
        assert 'target_column' in preprocessor.config
        assert preprocessor.config['target_column'] == 'nutrition_grade_fr'
        assert 'outlier_threshold' in preprocessor.config
    
    def test_custom_config(self):
        """Test that custom configuration overrides defaults."""
        custom_config = {
            'target_column': 'custom_grade',
            'outlier_threshold': 50
        }
        
        preprocessor = NutritionDataPreprocessor(custom_config)
        
        assert preprocessor.config['target_column'] == 'custom_grade'
        assert preprocessor.config['outlier_threshold'] == 50
        # Default values should still be present
        assert 'columns_to_drop' in preprocessor.config
    
    def test_load_data_csv(self, tmp_path):
        """Test loading CSV data."""
        # Create test CSV data
        test_data = pd.DataFrame({
            'fat_100g': [5, 10, 15],
            'carbohydrates_100g': [20, 30, 40],
            'nutrition_grade_fr': ['A', 'B', 'C']
        })
        
        csv_path = tmp_path / "test_data.csv"
        test_data.to_csv(csv_path, index=False)
        
        preprocessor = NutritionDataPreprocessor()
        loaded_data = preprocessor.load_data(csv_path)
        
        assert loaded_data.shape == (3, 3)
        assert 'fat_100g' in loaded_data.columns
    
    def test_load_data_parquet(self, tmp_path):
        """Test loading Parquet data."""
        # Create test Parquet data
        test_data = pd.DataFrame({
            'fat_100g': [5, 10, 15],
            'carbohydrates_100g': [20, 30, 40],
            'nutrition_grade_fr': ['A', 'B', 'C']
        })
        
        parquet_path = tmp_path / "test_data.parquet"
        test_data.to_parquet(parquet_path, index=False)
        
        preprocessor = NutritionDataPreprocessor()
        loaded_data = preprocessor.load_data(parquet_path)
        
        assert loaded_data.shape == (3, 3)
        assert 'fat_100g' in loaded_data.columns
    
    def test_sample_data(self):
        """Test data sampling functionality."""
        # Create test data
        test_data = pd.DataFrame({
            'fat_100g': range(100),
            'carbohydrates_100g': range(100),
            'nutrition_grade_fr': ['A'] * 50 + ['B'] * 50
        })
        
        config = {'sample_fraction': 0.5, 'random_state': 42}
        preprocessor = NutritionDataPreprocessor(config)
        
        sampled_data = preprocessor.sample_data(test_data)
        
        assert len(sampled_data) == 50
        assert sampled_data.index.tolist() == list(range(50))
    
    def test_remove_columns(self):
        """Test column removal functionality."""
        test_data = pd.DataFrame({
            'fat_100g': [5, 10, 15],
            'carbohydrates_100g': [20, 30, 40],
            'energy_100g': [100, 200, 300],  # Should be dropped
            'nutrition_grade_fr': ['A', 'B', 'C']
        })
        
        preprocessor = NutritionDataPreprocessor()
        cleaned_data = preprocessor.remove_columns(test_data)
        
        assert 'energy_100g' not in cleaned_data.columns
        assert 'fat_100g' in cleaned_data.columns
        assert cleaned_data.shape == (3, 3)
    
    def test_clean_target_variable(self):
        """Test target variable cleaning."""
        test_data = pd.DataFrame({
            'fat_100g': [5, 10, 15, 20],
            'nutrition_grade_fr': ['A', 'B', 'unknown', 'C']
        })
        
        preprocessor = NutritionDataPreprocessor()
        cleaned_data = preprocessor.clean_target_variable(test_data)
        
        assert len(cleaned_data) == 3  # 'unknown' should be removed
        assert 'unknown' not in cleaned_data['nutrition_grade_fr'].values
    
    def test_remove_outliers(self):
        """Test outlier removal functionality."""
        test_data = pd.DataFrame({
            'fat_100g': [5, 10, 150],  # 150 is outlier
            'carbohydrates_100g': [20, 30, 40],
            'nutrition_grade_fr': ['A', 'B', 'C']
        })
        
        preprocessor = NutritionDataPreprocessor()
        cleaned_data = preprocessor.remove_outliers(test_data)
        
        assert len(cleaned_data) == 2  # Outlier row should be removed
        assert 150 not in cleaned_data['fat_100g'].values
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        test_data = pd.DataFrame({
            'fat_100g': [5, np.nan, 15],
            'carbohydrates_100g': [20, 30, np.nan],
            'nutrition_grade_fr': ['A', 'B', 'C']
        })
        
        preprocessor = NutritionDataPreprocessor()
        cleaned_data = preprocessor.handle_missing_values(test_data)
        
        assert cleaned_data['fat_100g'].isna().sum() == 0
        assert cleaned_data['carbohydrates_100g'].isna().sum() == 0
    
    def test_create_features(self):
        """Test feature engineering functionality."""
        test_data = pd.DataFrame({
            'fat_100g': [5, 10, 15],
            'carbohydrates_100g': [20, 30, 40],
            'proteins_100g': [8, 12, 16],
            'nutrition_grade_fr': ['A', 'B', 'C']
        })
        
        preprocessor = NutritionDataPreprocessor()
        enhanced_data = preprocessor.create_features(test_data)
        
        # Check if ratio features were created
        assert 'fat_carb_ratio' in enhanced_data.columns
        assert 'protein_carb_ratio' in enhanced_data.columns
        assert 'total_macros' in enhanced_data.columns
    
    def test_prepare_for_modeling(self):
        """Test data preparation for modeling."""
        test_data = pd.DataFrame({
            'fat_100g': [5, 10, 15],
            'carbohydrates_100g': [20, 30, 40],
            'nutrition_grade_fr': ['A', 'B', 'C']
        })
        
        preprocessor = NutritionDataPreprocessor()
        features_df, feature_columns, target_col = preprocessor.prepare_for_modeling(test_data)
        
        assert target_col == 'nutrition_grade_fr'
        assert len(feature_columns) == 2
        assert 'fat_100g' in feature_columns
        assert 'carbohydrates_100g' in feature_columns
        assert features_df.shape == (3, 2)
    
    def test_complete_preprocessing_pipeline(self, tmp_path):
        """Test the complete preprocessing pipeline."""
        # Create test data
        test_data = pd.DataFrame({
            'fat_100g': [5, 10, 15, 20],
            'carbohydrates_100g': [20, 30, 40, 50],
            'proteins_100g': [8, 12, 16, 20],
            'energy_100g': [100, 200, 300, 400],  # Should be dropped
            'nutrition_grade_fr': ['A', 'B', 'unknown', 'C']
        })
        
        # Save test data
        data_path = tmp_path / "test_data.parquet"
        test_data.to_parquet(data_path, index=False)
        
        # Run preprocessing
        preprocessor = NutritionDataPreprocessor()
        features_df, target_df, feature_columns, target_col = preprocessor.preprocess(
            data_path, save_intermediate=False
        )
        
        # Check results
        assert target_col == 'nutrition_grade_fr'
        assert len(feature_columns) >= 2  # Should have at least fat and carbs
        assert features_df.shape[0] == 3  # 'unknown' should be removed
        assert target_df.shape == (3, 1)


if __name__ == "__main__":
    pytest.main([__file__]) 