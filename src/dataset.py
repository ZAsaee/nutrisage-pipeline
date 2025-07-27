"""
Data loading and processing utilities for NutriSage nutrition grade prediction.
Handles data loading from various sources (S3, local files) and initial data validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union
import boto3
import awswrangler as wr
from loguru import logger
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR

app = typer.Typer()


def load_from_s3(bucket: str, key: str, sample_fraction: float = None) -> pd.DataFrame:
    """Load partitioned parquet data from S3."""
    s3_path = f"s3://{bucket}/{key}"
    
    # Read all partitions first
    df = wr.s3.read_parquet(s3_path)
    
    # Apply sampling if requested
    if sample_fraction:
        df = sample_data(df, sample_fraction, random_state=42)
    
    return df


def load_from_local(file_path: Path) -> pd.DataFrame:
    """
    Load data from local file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading data from {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        if file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Successfully loaded data: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from local file: {str(e)}")
        raise


def validate_nutrition_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate nutrition data for required columns and data quality.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating nutrition data...")
    
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'data_quality_issues': [],
        'recommendations': []
    }
    
    # Required columns for nutrition grade prediction
    required_columns = [
        'fat_100g',
        'carbohydrates_100g', 
        'proteins_100g',
        'nutrition_grade_fr'
    ]
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_results['missing_columns'] = missing_columns
        validation_results['is_valid'] = False
        validation_results['recommendations'].append(f"Missing required columns: {missing_columns}")
    
    # Check data quality
    if len(df) == 0:
        validation_results['data_quality_issues'].append("Dataset is empty")
        validation_results['is_valid'] = False
    
    # Check for missing values in key columns
    if 'nutrition_grade_fr' in df.columns:
        missing_grades = df['nutrition_grade_fr'].isna().sum()
        if missing_grades > 0:
            validation_results['data_quality_issues'].append(f"{missing_grades} missing nutrition grades")
    
    # Check for extreme outliers
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in df.columns:
            q99 = df[col].quantile(0.99)
            extreme_outliers = (df[col] > q99 * 10).sum()
            if extreme_outliers > 0:
                validation_results['data_quality_issues'].append(f"{extreme_outliers} extreme outliers in {col}")
    
    # Log validation results
    if validation_results['is_valid']:
        logger.success("Data validation passed!")
    else:
        logger.warning("Data validation issues found:")
        for issue in validation_results['data_quality_issues']:
            logger.warning(f"  - {issue}")
        for rec in validation_results['recommendations']:
            logger.info(f"  - {rec}")
    
    return validation_results


def sample_data(df: pd.DataFrame, sample_fraction: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """
    Sample data for development and testing.
    
    Args:
        df: Input DataFrame
        sample_fraction: Fraction of data to sample
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame
    """
    logger.info(f"Sampling {sample_fraction*100}% of data")
    
    sampled_df = df.sample(frac=sample_fraction, random_state=random_state)
    sampled_df.reset_index(drop=True, inplace=True)
    
    logger.info(f"Sampled data shape: {sampled_df.shape}")
    return sampled_df


def save_data(df: pd.DataFrame, output_path: Path, file_format: str = 'parquet') -> None:
    """
    Save data to specified location and format.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
        file_format: File format ('parquet', 'csv', 'json')
    """
    logger.info(f"Saving data to {output_path}")
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if file_format.lower() == 'parquet':
            df.to_parquet(output_path, index=False)
        elif file_format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif file_format.lower() == 'json':
            df.to_json(output_path, orient='records')
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.success(f"Data saved successfully to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of the dataset.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with data summary
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    # Add target variable summary if available
    if 'nutrition_grade_fr' in df.columns:
        summary['target_distribution'] = df['nutrition_grade_fr'].value_counts().to_dict()
    
    return summary


@app.command()
def main(
    input_source: str = typer.Option("local", "--source", "-s", help="Data source: 'local' or 's3'"),
    input_path: str = typer.Option("", "--input", "-i", help="Input file path or S3 URI"),
    output_path: Path = typer.Option(PROCESSED_DATA_DIR / "raw_data.parquet", "--output", "-o", help="Output file path"),
    sample_fraction: Optional[float] = typer.Option(None, "--sample", help="Fraction of data to sample (0.0-1.0)"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Validate data quality"),
    file_format: str = typer.Option("parquet", "--format", "-f", help="Output file format"),
    random_state: int = typer.Option(42, "--random-state", help="Random seed for sampling")
):
    """
    Load and process nutrition data from various sources.
    
    Examples:
        # Load from local file
        python -m src.dataset --input data/raw/nutrition_data.csv
        
        # Load from S3
        python -m src.dataset --source s3 --input s3://nutrisage/data/nutrition_data.parquet
        
        # Sample 10% of data
        python -m src.dataset --input data.csv --sample 0.1
    """
    logger.info("Starting data loading and processing...")
    
    try:
        # Load data based on source
        if input_source.lower() == 's3':
            if not input_path:
                raise ValueError("S3 URI must be provided when source is 's3'")
            
            # Parse S3 URI
            if input_path.startswith('s3://'):
                s3_uri = input_path
                bucket = s3_uri.split('/')[2]
                key = '/'.join(s3_uri.split('/')[3:])
            else:
                raise ValueError("Invalid S3 URI format. Use: s3://bucket/key")
            
            df = load_from_s3(bucket, key, sample_fraction)
            
        else:  # local
            if not input_path:
                raise ValueError("Input file path must be provided")
            
            df = load_from_local(Path(input_path))
        
        # Validate data if requested
        if validate:
            validation_results = validate_nutrition_data(df)
            if not validation_results['is_valid']:
                logger.warning("Data validation failed, but continuing...")
        
        # Sample data if requested
        if sample_fraction:
            df = sample_data(df, sample_fraction, random_state)
        
        # Generate and log summary
        summary = get_data_summary(df)
        logger.info(f"Data summary:")
        logger.info(f"  Shape: {summary['shape']}")
        logger.info(f"  Columns: {len(summary['columns'])}")
        logger.info(f"  Numeric columns: {len(summary['numeric_columns'])}")
        logger.info(f"  Categorical columns: {len(summary['categorical_columns'])}")
        
        if 'target_distribution' in summary:
            logger.info(f"  Target distribution: {summary['target_distribution']}")
        
        # Save processed data
        save_data(df, output_path, file_format)
        
        logger.success("Data loading and processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        raise


if __name__ == "__main__":
    app()
