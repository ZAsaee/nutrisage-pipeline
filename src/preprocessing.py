"""
Data preprocessing module for NutriSage nutrition grade prediction.
Implements data cleaning and feature engineering steps from the notebook.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import typer

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR


class NutritionDataPreprocessor:
    """Preprocessor for nutrition data cleaning and feature engineering."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or self._get_default_config()
        self.feature_columns = None
        self.target_column = None
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration."""
        return {
            'columns_to_drop': [
                'energy_100g', 'saturated-fat_100g', 'fiber_100g', 'main_category',
                'categories_tags', 'brands_tags', 'countries_tags', 'serving_size',
                'created_t', 'year', 'country'
            ],
            'target_column': 'nutrition_grade_fr',
            'invalid_grades': ['unknown', 'not-applicable'],
            'outlier_threshold': 100,
            'outlier_columns': ['fat_100g', 'carbohydrates_100g', 'proteins_100g', 'salt_100g', 'sugars_100g'],
            'sample_fraction': None,  # Set to a value like 0.1 for sampling
            'random_state': 42
        }
    
    def load_data(self, data_path: Path) -> pd.DataFrame:
        """
        Load data from various formats (CSV, Parquet, etc.).
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        
        if data_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(data_path)
        elif data_path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        logger.info(f"Loaded data shape: {df.shape}")
        return df
    
    def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample data if specified in config.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Sampled DataFrame
        """
        if self.config['sample_fraction']:
            logger.info(f"Sampling {self.config['sample_fraction']*100}% of data")
            df = df.sample(frac=self.config['sample_fraction'], 
                          random_state=self.config['random_state'])
            df.reset_index(drop=True, inplace=True)
            logger.info(f"Sampled data shape: {df.shape}")
        
        return df
    
    def remove_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove specified columns from the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with columns removed
        """
        logger.info("Removing specified columns")
        columns_to_drop = [col for col in self.config['columns_to_drop'] if col in df.columns]
        
        if columns_to_drop:
            logger.info(f"Dropping columns: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
            logger.info(f"Data shape after column removal: {df.shape}")
        else:
            logger.info("No columns to drop found in dataset")
        
        return df
    
    def clean_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the target variable by removing invalid grades.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned target variable
        """
        target_col = self.config['target_column']
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        logger.info(f"Cleaning target variable: {target_col}")
        
        # Remove invalid grades
        initial_count = len(df)
        df = df.loc[~df[target_col].isin(self.config['invalid_grades']), :]
        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} rows with invalid grades")
        
        # Remove rows with missing target values
        initial_count = len(df)
        df = df.loc[~df[target_col].isna(), :]
        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} rows with missing target values")
        
        logger.info(f"Target variable distribution:\n{df[target_col].value_counts()}")
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from numeric columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers removed
        """
        logger.info("Removing outliers")
        
        outlier_columns = [col for col in self.config['outlier_columns'] if col in df.columns]
        threshold = self.config['outlier_threshold']
        
        initial_count = len(df)
        
        for col in outlier_columns:
            df = df.loc[~(df[col] > threshold), :]
        
        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} outlier rows")
        logger.info(f"Data shape after outlier removal: {df.shape}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
            
            # For numeric columns, fill with median
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled missing values in {col} with median: {median_val}")
        else:
            logger.info("No missing values found")
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features if needed.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        logger.info("Creating additional features")
        
        # Example: Create ratio features
        if 'fat_100g' in df.columns and 'carbohydrates_100g' in df.columns:
            df['fat_carb_ratio'] = df['fat_100g'] / (df['carbohydrates_100g'] + 1e-8)
            logger.info("Created fat_carb_ratio feature")
        
        if 'proteins_100g' in df.columns and 'carbohydrates_100g' in df.columns:
            df['protein_carb_ratio'] = df['proteins_100g'] / (df['carbohydrates_100g'] + 1e-8)
            logger.info("Created protein_carb_ratio feature")
        
        # Example: Create total macronutrient feature
        macro_cols = ['fat_100g', 'carbohydrates_100g', 'proteins_100g']
        existing_macro_cols = [col for col in macro_cols if col in df.columns]
        if len(existing_macro_cols) >= 2:
            df['total_macros'] = df[existing_macro_cols].sum(axis=1)
            logger.info("Created total_macros feature")
        
        return df
    
    def prepare_for_modeling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
        """
        Prepare data for modeling by separating features and target.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features_df, feature_columns, target_column)
        """
        target_col = self.config['target_column']
        
        # Get feature columns (all numeric columns except target)
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in feature_columns:
            feature_columns.remove(target_col)
        
        # Create features DataFrame
        features_df = df[feature_columns].copy()
        
        # Create target DataFrame
        target_df = df[[target_col]].copy()
        
        logger.info(f"Prepared {len(feature_columns)} features for modeling")
        logger.info(f"Feature columns: {feature_columns}")
        
        return features_df, feature_columns, target_col
    
    def preprocess(self, data_path: Path, save_intermediate: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
        """
        Complete preprocessing pipeline.
        
        Args:
            data_path: Path to input data
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Tuple of (features_df, target_df, feature_columns, target_column)
        """
        logger.info("Starting data preprocessing pipeline")
        
        # Load data
        df = self.load_data(data_path)
        
        # Sample data if specified
        df = self.sample_data(df)
        
        # Save raw sample if sampling was done
        if save_intermediate and self.config['sample_fraction']:
            sample_path = RAW_DATA_DIR / "sample_data.parquet"
            df.to_parquet(sample_path)
            logger.info(f"Saved sample data to {sample_path}")
        
        # Remove columns
        df = self.remove_columns(df)
        
        # Clean target variable
        df = self.clean_target_variable(df)
        
        # Remove outliers
        df = self.remove_outliers(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create additional features
        df = self.create_features(df)
        
        # Save cleaned data
        if save_intermediate:
            clean_path = PROCESSED_DATA_DIR / "clean_data.parquet"
            df.to_parquet(clean_path)
            logger.info(f"Saved cleaned data to {clean_path}")
        
        # Prepare for modeling
        features_df, feature_columns, target_col = self.prepare_for_modeling(df)
        target_df = df[[target_col]]
        
        # Save features and target separately
        if save_intermediate:
            features_path = PROCESSED_DATA_DIR / "features.csv"
            target_path = PROCESSED_DATA_DIR / "labels.csv"
            
            features_df.to_csv(features_path, index=False)
            target_df.to_csv(target_path, index=False)
            
            logger.info(f"Saved features to {features_path}")
            logger.info(f"Saved target to {target_path}")
        
        logger.success("Preprocessing pipeline completed successfully!")
        
        return features_df, target_df, feature_columns, target_col


# Command-line interface
app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "data.parquet",
    config_path: Optional[Path] = None,
    save_intermediate: bool = True,
    sample_fraction: Optional[float] = None,
    random_state: int = 42
):
    """
    Run the complete preprocessing pipeline.
    
    Args:
        input_path: Path to input data file
        config_path: Path to configuration file (optional)
        save_intermediate: Whether to save intermediate results
        sample_fraction: Fraction of data to sample (optional)
        random_state: Random state for sampling
    """
    logger.info("Starting NutriSage data preprocessing")
    
    try:
        # Load configuration if provided
        config = None
        if config_path and config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Override config with command line arguments
        if sample_fraction is not None:
            if config is None:
                config = {}
            config['sample_fraction'] = sample_fraction
            config['random_state'] = random_state
        
        # Initialize preprocessor
        preprocessor = NutritionDataPreprocessor(config)
        
        # Run preprocessing
        features_df, target_df, feature_columns, target_col = preprocessor.preprocess(
            input_path, save_intermediate
        )
        
        logger.info(f"Preprocessing completed. Final shapes:")
        logger.info(f"  Features: {features_df.shape}")
        logger.info(f"  Target: {target_df.shape}")
        logger.info(f"  Feature columns: {len(feature_columns)}")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    app() 