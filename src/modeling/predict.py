"""
Prediction module for nutrition grade classification.

This module provides functionality to load trained models and make predictions
on new nutrition data, with support for both single predictions and batch processing.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
import json

from loguru import logger
from tqdm import tqdm
import typer
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from src.config import (
    MODELS_DIR, 
    PROCESSED_DATA_DIR, 
    RAW_DATA_DIR,
    get_model_info,
    ensure_directories
)
from src.dataset import load_from_local, load_from_s3, validate_nutrition_data
from src.preprocessing import NutritionDataPreprocessor

app = typer.Typer()


def load_model_and_metadata(model_path: Path, metadata_path: Path) -> tuple:
    """
    Load trained model and metadata.
    
    Args:
        model_path: Path to the trained model file
        metadata_path: Path to the model metadata file
        
    Returns:
        Tuple of (model, metadata)
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Model type: {metadata.get('model_type', 'unknown')}")
        logger.info(f"Number of classes: {metadata.get('num_classes', 'unknown')}")
        logger.info(f"Classes: {metadata.get('classes', [])}")
        logger.info(f"Number of features: {len(metadata.get('feature_columns', []))}")
        
        return model, metadata
        
    except FileNotFoundError as e:
        logger.error(f"Model or metadata file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def preprocess_input_data(
    data: pd.DataFrame, 
    metadata: Dict[str, Any],
    preprocessor: Optional[NutritionDataPreprocessor] = None
) -> pd.DataFrame:
    """
    Preprocess input data to match training data format.
    
    Args:
        data: Input DataFrame
        metadata: Model metadata containing feature information
        preprocessor: Optional preprocessor instance for raw data
        
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    logger.info("Preprocessing input data for prediction")
    
    # Get expected feature columns from metadata
    expected_features = metadata.get('feature_columns', [])
    
    if not expected_features:
        raise ValueError("No feature columns found in model metadata")
    
    # Check if all expected features are present
    missing_features = set(expected_features) - set(data.columns)
    if missing_features:
        logger.warning(f"Missing features in input data: {missing_features}")
        logger.info("Attempting to create missing features...")
        
        # Try to create missing features using basic calculations
        if 'fat_carb_ratio' in missing_features and 'fat_100g' in data.columns and 'carbohydrates_100g' in data.columns:
            data['fat_carb_ratio'] = data['fat_100g'] / (data['carbohydrates_100g'] + 1e-8)
        
        if 'protein_carb_ratio' in missing_features and 'proteins_100g' in data.columns and 'carbohydrates_100g' in data.columns:
            data['protein_carb_ratio'] = data['proteins_100g'] / (data['carbohydrates_100g'] + 1e-8)
        
        if 'protein_fat_ratio' in missing_features and 'proteins_100g' in data.columns and 'fat_100g' in data.columns:
            data['protein_fat_ratio'] = data['proteins_100g'] / (data['fat_100g'] + 1e-8)
        
        if 'total_macros' in missing_features:
            macro_cols = ['fat_100g', 'carbohydrates_100g', 'proteins_100g']
            available_macros = [col for col in macro_cols if col in data.columns]
            if available_macros:
                data['total_macros'] = data[available_macros].sum(axis=1)
        
        # Check again for missing features
        missing_features = set(expected_features) - set(data.columns)
        if missing_features:
            raise ValueError(f"Still missing features after creation: {missing_features}")
    
    # Select only the expected features
    X = data[expected_features]
    
    # Ensure correct data types
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing values in input data")
        # Fill with median for numeric columns
        X = X.fillna(X.median())
    
    logger.info(f"Preprocessed data shape: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")
    
    return X


def make_predictions(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    metadata: Dict[str, Any],
    return_probabilities: bool = True
) -> Dict[str, Any]:
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained XGBoost model
        X: Preprocessed feature data
        metadata: Model metadata
        return_probabilities: Whether to return prediction probabilities
        
    Returns:
        Dictionary containing predictions and metadata
    """
    logger.info(f"Making predictions on {len(X)} samples")
    
    try:
        # Make predictions
        predictions = model.predict(X)
        
        # Get prediction probabilities if requested
        probabilities = None
        if return_probabilities:
            probabilities = model.predict_proba(X)
        
        # Get label encoder for class mapping
        label_encoder = metadata.get('label_encoder')
        classes = metadata.get('classes', [])
        
        # Convert numeric predictions to class labels if encoder available
        if label_encoder is not None:
            predicted_labels = label_encoder.inverse_transform(predictions)
        else:
            predicted_labels = predictions
        
        # Create results dictionary
        results = {
            'predictions': predictions,
            'predicted_labels': predicted_labels,
            'probabilities': probabilities,
            'classes': classes,
            'num_samples': len(X)
        }
        
        # Log prediction summary
        if label_encoder is not None:
            unique_predictions, counts = np.unique(predicted_labels, return_counts=True)
            logger.info("Prediction distribution:")
            for pred, count in zip(unique_predictions, counts):
                logger.info(f"  {pred}: {count} ({count/len(predictions)*100:.1f}%)")
        
        logger.success(f"Predictions completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise


def save_predictions(
    results: Dict[str, Any],
    output_path: Path,
    input_data: Optional[pd.DataFrame] = None,
    include_probabilities: bool = True,
    include_features: bool = False
) -> None:
    """
    Save prediction results to file.
    
    Args:
        results: Prediction results from make_predictions
        output_path: Path to save predictions
        input_data: Original input data (optional)
        include_probabilities: Whether to include prediction probabilities
        include_features: Whether to include input features in output
    """
    logger.info(f"Saving predictions to {output_path}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare output DataFrame
    output_data = []
    
    for i in range(results['num_samples']):
        row = {
            'sample_id': i,
            'predicted_class': results['predicted_labels'][i],
            'predicted_class_numeric': results['predictions'][i]
        }
        
        # Add probabilities if available
        if include_probabilities and results['probabilities'] is not None:
            for j, class_name in enumerate(results['classes']):
                row[f'prob_{class_name}'] = results['probabilities'][i][j]
        
        # Add input features if requested
        if include_features and input_data is not None:
            for col in input_data.columns:
                row[f'feature_{col}'] = input_data.iloc[i][col]
        
        output_data.append(row)
    
    # Create DataFrame and save
    output_df = pd.DataFrame(output_data)
    
    # Determine file format and save
    if output_path.suffix.lower() == '.csv':
        output_df.to_csv(output_path, index=False)
    elif output_path.suffix.lower() == '.parquet':
        output_df.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == '.json':
        output_df.to_json(output_path, orient='records', indent=2)
    else:
        # Default to CSV
        output_path = output_path.with_suffix('.csv')
        output_df.to_csv(output_path, index=False)
    
    logger.success(f"Predictions saved to {output_path}")
    logger.info(f"Output shape: {output_df.shape}")


def predict_single_sample(
    model: xgb.XGBClassifier,
    metadata: Dict[str, Any],
    sample_data: Dict[str, Any],
    preprocessor: Optional[NutritionDataPreprocessor] = None
) -> Dict[str, Any]:
    """
    Make prediction for a single sample.
    
    Args:
        model: Trained model
        metadata: Model metadata
        sample_data: Dictionary containing sample features
        preprocessor: Optional preprocessor for raw data
        
    Returns:
        Prediction results for single sample
    """
    # Convert to DataFrame
    sample_df = pd.DataFrame([sample_data])
    
    # Preprocess
    X = preprocess_input_data(sample_df, metadata, preprocessor)
    
    # Make prediction
    results = make_predictions(model, X, metadata, return_probabilities=True)
    
    # Return single sample results
    return {
        'predicted_class': results['predicted_labels'][0],
        'predicted_class_numeric': results['predictions'][0],
        'probabilities': results['probabilities'][0] if results['probabilities'] is not None else None,
        'classes': results['classes']
    }


@app.command()
def main(
    input_path: Path = typer.Option(RAW_DATA_DIR / "data.parquet", "--input", "-i", help="Input data path or S3 URI"),
    model_path: Path = typer.Option(MODELS_DIR / "nutrition_grade_model.pkl", "--model", "-m", help="Path to trained model"),
    metadata_path: Path = typer.Option(MODELS_DIR / "model_metadata.pkl", "--metadata", help="Path to model metadata"),
    output_path: Path = typer.Option(PROCESSED_DATA_DIR / "predictions.csv", "--output", "-o", help="Output predictions path"),
    input_source: str = typer.Option("local", "--source", "-s", help="Input data source: 'local' or 's3'"),
    use_preprocessing: bool = typer.Option(True, "--preprocess/--no-preprocess", help="Use preprocessing pipeline for raw data"),
    include_probabilities: bool = typer.Option(True, "--probs/--no-probs", help="Include prediction probabilities in output"),
    include_features: bool = typer.Option(False, "--features/--no-features", help="Include input features in output"),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to preprocessing config file"),
    sample_fraction: Optional[float] = typer.Option(None, "--sample", help="Fraction of data to sample for prediction")
):
    """
    Make predictions using trained nutrition grade model.
    
    Examples:
        # Predict on local preprocessed data
        python -m src.modeling.predict --input data/processed/features.csv
        
        # Predict on raw data using preprocessing pipeline
        python -m src.modeling.predict --input data/raw/nutrition_data.parquet --preprocess
        
        # Predict on S3 data
        python -m src.modeling.predict --source s3 --input s3://bucket/data.parquet
        
        # Save with probabilities
        python -m src.modeling.predict --input data.csv --probs --output predictions.csv
    """
    logger.info("Starting nutrition grade prediction...")
    
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Load model and metadata
        model, metadata = load_model_and_metadata(model_path, metadata_path)
        
        # Load input data
        logger.info(f"Loading input data from {input_source}: {input_path}")
        
        if input_source.lower() == 's3':
            # Parse S3 URI
            if str(input_path).startswith('s3://'):
                s3_uri = str(input_path)
                bucket = s3_uri.split('/')[2]
                key = '/'.join(s3_uri.split('/')[3:])
            else:
                raise ValueError("Invalid S3 URI format. Use: s3://bucket/key")
            
            data = load_from_s3(bucket, key)
        else:
            data = load_from_local(input_path)
        
        # Validate data
        validation_results = validate_nutrition_data(data)
        if not validation_results['is_valid']:
            logger.warning("Data validation issues found, but continuing with prediction...")
        
        # Sample data if requested
        if sample_fraction:
            logger.info(f"Sampling {sample_fraction*100}% of data")
            data = data.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
        
        logger.info(f"Input data shape: {data.shape}")
        
        # Initialize preprocessor if needed
        preprocessor = None
        if use_preprocessing:
            logger.info("Initializing preprocessing pipeline")
            preprocessor = NutritionDataPreprocessor(config_path)
        
        # Preprocess data
        X = preprocess_input_data(data, metadata, preprocessor)
        
        # Make predictions
        results = make_predictions(model, X, metadata, return_probabilities=include_probabilities)
        
        # Save predictions
        save_predictions(
            results, 
            output_path, 
            input_data=data if include_features else None,
            include_probabilities=include_probabilities,
            include_features=include_features
        )
        
        # Print summary
        logger.success("Prediction pipeline completed successfully!")
        logger.info(f"Processed {results['num_samples']} samples")
        logger.info(f"Predictions saved to: {output_path}")
        
        # Show prediction distribution
        if 'predicted_labels' in results:
            unique_predictions, counts = np.unique(results['predicted_labels'], return_counts=True)
            logger.info("Prediction distribution:")
            for pred, count in zip(unique_predictions, counts):
                logger.info(f"  {pred}: {count} ({count/results['num_samples']*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        raise


if __name__ == "__main__":
    app()
