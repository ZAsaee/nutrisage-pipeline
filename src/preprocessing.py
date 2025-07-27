"""
Data preprocessing module for NutriSage nutrition grade prediction.
Implements data cleaning and feature engineering steps from the notebook.
Integrates with dataset.py for data loading and validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import typer
import json

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.dataset import load_from_local, load_from_s3, validate_nutrition_data, save_data


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
        self.preprocessing_stats = {}
        
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
            'random_state': 42,
            'feature_engineering': {
                'create_ratios': True,
                'create_totals': True,
                'create_interactions': False
            },
            'missing_value_strategy': {
                'numeric': 'median',
                'categorical': 'mode'
            }
        }
    
    def load_data(self, data_path: Path, source: str = 'local') -> pd.DataFrame:
        """
        Load data using the dataset module functionality.
        
        Args:
            data_path: Path to the data file or S3 URI
            source: Data source ('local' or 's3')
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {source}: {data_path}")
        
        try:
            if source.lower() == 's3':
                # Parse S3 URI
                if str(data_path).startswith('s3://'):
                    s3_uri = str(data_path)
                    bucket = s3_uri.split('/')[2]
                    key = '/'.join(s3_uri.split('/')[3:])
                else:
                    raise ValueError("Invalid S3 URI format. Use: s3://bucket/key")
                
                df = load_from_s3(bucket, key)
            else:
                df = load_from_local(data_path)
            
            # Validate the loaded data
            validation_results = validate_nutrition_data(df)
            if not validation_results['is_valid']:
                logger.warning("Data validation issues found, but continuing with preprocessing...")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
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
            initial_count = len(df)
            df = df.sample(frac=self.config['sample_fraction'], 
                          random_state=self.config['random_state'])
            df.reset_index(drop=True, inplace=True)
            final_count = len(df)
            logger.info(f"Sampled data: {initial_count} -> {final_count} rows")
            
            # Store sampling stats
            self.preprocessing_stats['sampling'] = {
                'initial_count': initial_count,
                'final_count': final_count,
                'sample_fraction': self.config['sample_fraction']
            }
        
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
        initial_columns = list(df.columns)
        columns_to_drop = [col for col in self.config['columns_to_drop'] if col in df.columns]
        
        if columns_to_drop:
            logger.info(f"Dropping columns: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
            final_columns = list(df.columns)
            logger.info(f"Columns: {len(initial_columns)} -> {len(final_columns)}")
            
            # Store column removal stats
            self.preprocessing_stats['column_removal'] = {
                'removed_columns': columns_to_drop,
                'initial_columns': len(initial_columns),
                'final_columns': len(final_columns)
            }
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
        
        initial_count = len(df)
        
        # Remove invalid grades
        df = df.loc[~df[target_col].isin(self.config['invalid_grades']), :]
        after_invalid_removal = len(df)
        
        # Remove rows with missing target values
        df = df.loc[~df[target_col].isna(), :]
        final_count = len(df)
        
        removed_invalid = initial_count - after_invalid_removal
        removed_missing = after_invalid_removal - final_count
        
        logger.info(f"Target cleaning: {initial_count} -> {final_count} rows")
        logger.info(f"  Removed {removed_invalid} invalid grades")
        logger.info(f"  Removed {removed_missing} missing grades")
        logger.info(f"Target distribution:\n{df[target_col].value_counts()}")
        
        # Store target cleaning stats
        self.preprocessing_stats['target_cleaning'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_invalid': removed_invalid,
            'removed_missing': removed_missing,
            'target_distribution': df[target_col].value_counts().to_dict()
        }
        
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
        outlier_stats = {}
        
        for col in outlier_columns:
            outliers_mask = df[col] > threshold
            outlier_count = outliers_mask.sum()
            if outlier_count > 0:
                df = df.loc[~outliers_mask, :]
                outlier_stats[col] = outlier_count
                logger.info(f"  Removed {outlier_count} outliers from {col}")
        
        final_count = len(df)
        total_removed = initial_count - final_count
        
        if total_removed > 0:
            logger.info(f"Outlier removal: {initial_count} -> {final_count} rows ({total_removed} removed)")
        else:
            logger.info("No outliers found")
        
        # Store outlier removal stats
        self.preprocessing_stats['outlier_removal'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'total_removed': total_removed,
            'outlier_counts': outlier_stats
        }
        
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
        
        missing_stats = {}
        strategy = self.config['missing_value_strategy']
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0]
        
        if len(columns_with_missing) > 0:
            logger.info(f"Missing values found in {len(columns_with_missing)} columns")
            
            for col in columns_with_missing.index:
                missing_count = missing_counts[col]
                if missing_count > 0:
                    if df[col].dtype in ['int64', 'float64']:
                        # Numeric columns
                        if strategy['numeric'] == 'median':
                            fill_value = df[col].median()
                        elif strategy['numeric'] == 'mean':
                            fill_value = df[col].mean()
                        else:
                            fill_value = 0
                        
                        df[col] = df[col].fillna(fill_value)
                        missing_stats[col] = {
                            'type': 'numeric',
                            'missing_count': missing_count,
                            'fill_value': fill_value,
                            'strategy': strategy['numeric']
                        }
                        logger.info(f"  Filled {missing_count} missing values in {col} with {strategy['numeric']}: {fill_value}")
                    else:
                        # Categorical columns
                        if strategy['categorical'] == 'mode':
                            fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                        else:
                            fill_value = 'Unknown'
                        
                        df[col] = df[col].fillna(fill_value)
                        missing_stats[col] = {
                            'type': 'categorical',
                            'missing_count': missing_count,
                            'fill_value': fill_value,
                            'strategy': strategy['categorical']
                        }
                        logger.info(f"  Filled {missing_count} missing values in {col} with {strategy['categorical']}: {fill_value}")
        else:
            logger.info("No missing values found")
        
        # Store missing value handling stats
        self.preprocessing_stats['missing_value_handling'] = missing_stats
        
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
        
        feature_engineering_config = self.config['feature_engineering']
        created_features = []
        
        # Create ratio features
        if feature_engineering_config['create_ratios']:
            if 'fat_100g' in df.columns and 'carbohydrates_100g' in df.columns:
                df['fat_carb_ratio'] = df['fat_100g'] / (df['carbohydrates_100g'] + 1e-8)
                created_features.append('fat_carb_ratio')
                logger.info("Created fat_carb_ratio feature")
            
            if 'proteins_100g' in df.columns and 'carbohydrates_100g' in df.columns:
                df['protein_carb_ratio'] = df['proteins_100g'] / (df['carbohydrates_100g'] + 1e-8)
                created_features.append('protein_carb_ratio')
                logger.info("Created protein_carb_ratio feature")
            
            if 'proteins_100g' in df.columns and 'fat_100g' in df.columns:
                df['protein_fat_ratio'] = df['proteins_100g'] / (df['fat_100g'] + 1e-8)
                created_features.append('protein_fat_ratio')
                logger.info("Created protein_fat_ratio feature")
        
        # Create total macronutrient feature
        if feature_engineering_config['create_totals']:
            macro_cols = ['fat_100g', 'carbohydrates_100g', 'proteins_100g']
            existing_macro_cols = [col for col in macro_cols if col in df.columns]
            if len(existing_macro_cols) >= 2:
                df['total_macros'] = df[existing_macro_cols].sum(axis=1)
                created_features.append('total_macros')
                logger.info("Created total_macros feature")
        
        # Create interaction features
        if feature_engineering_config['create_interactions']:
            if 'fat_100g' in df.columns and 'sugars_100g' in df.columns:
                df['fat_sugars_interaction'] = df['fat_100g'] * df['sugars_100g']
                created_features.append('fat_sugars_interaction')
                logger.info("Created fat_sugars_interaction feature")
        
        # Store feature engineering stats
        self.preprocessing_stats['feature_engineering'] = {
            'created_features': created_features,
            'total_features_created': len(created_features)
        }
        
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
        
        # Store modeling preparation stats
        self.preprocessing_stats['modeling_preparation'] = {
            'feature_count': len(feature_columns),
            'feature_columns': feature_columns,
            'target_column': target_col,
            'final_shape': features_df.shape
        }
        
        return features_df, feature_columns, target_col
    
    def save_preprocessing_stats(self, output_path: Path) -> None:
        """
        Save preprocessing statistics to a JSON file.
        
        Args:
            output_path: Path to save the stats file
        """
        stats_path = output_path.parent / "preprocessing_stats.json"
        
        with open(stats_path, 'w') as f:
            json.dump(self.preprocessing_stats, f, indent=2, default=str)
        
        logger.info(f"Preprocessing statistics saved to {stats_path}")
    
    def preprocess(self, data_path: Path, source: str = 'local', save_intermediate: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
        """
        Complete preprocessing pipeline.
        
        Args:
            data_path: Path to input data
            source: Data source ('local' or 's3')
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Tuple of (features_df, target_df, feature_columns, target_column)
        """
        logger.info("Starting data preprocessing pipeline")
        
        # Load data
        df = self.load_data(data_path, source)
        
        # Sample data if specified
        df = self.sample_data(df)
        
        # Save raw sample if sampling was done
        if save_intermediate and self.config['sample_fraction']:
            sample_path = RAW_DATA_DIR / "sample_data.parquet"
            save_data(df, sample_path, 'parquet')
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
            save_data(df, clean_path, 'parquet')
            logger.info(f"Saved cleaned data to {clean_path}")
        
        # Prepare for modeling
        features_df, feature_columns, target_col = self.prepare_for_modeling(df)
        target_df = df[[target_col]]
        
        # Save features and target separately
        if save_intermediate:
            features_path = PROCESSED_DATA_DIR / "features.csv"
            target_path = PROCESSED_DATA_DIR / "labels.csv"
            
            save_data(features_df, features_path, 'csv')
            save_data(target_df, target_path, 'csv')
            
            logger.info(f"Saved features to {features_path}")
            logger.info(f"Saved target to {target_path}")
            
            # Save preprocessing statistics
            self.save_preprocessing_stats(features_path)
        
        logger.success("Preprocessing pipeline completed successfully!")
        
        return features_df, target_df, feature_columns, target_col


# Command-line interface
app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "data.parquet",
    source: str = typer.Option("local", "--source", "-s", help="Data source: 'local' or 's3'"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    save_intermediate: bool = typer.Option(True, "--save-intermediate/--no-save", help="Save intermediate results"),
    sample_fraction: Optional[float] = typer.Option(None, "--sample", help="Fraction of data to sample (0.0-1.0)"),
    random_state: int = typer.Option(42, "--random-state", help="Random seed for sampling")
):
    """
    Run the complete preprocessing pipeline.
    
    Examples:
        # Preprocess with default settings
        python -m src.preprocessing
        
        # Preprocess with custom config
        python -m src.preprocessing --config config/preprocessing_config.json
        
        # Preprocess S3 data with sampling
        python -m src.preprocessing --source s3 --input s3://bucket/data.parquet --sample 0.1
    """
    logger.info("Starting NutriSage data preprocessing")
    
    try:
        # Load configuration if provided
        config = None
        if config_path and config_path.exists():
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
            input_path, source, save_intermediate
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