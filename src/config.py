"""
Configuration settings for NutriSage nutrition grade prediction project.
Centralized configuration for paths, AWS settings, and model parameters.
"""

from pathlib import Path
import os
from typing import Dict, Any

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Project root directory
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"Project root: {PROJ_ROOT}")

# Data directories
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model and output directories
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# =============================================================================
# AWS CONFIGURATION
# =============================================================================

# AWS settings
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "nutrisage")
AWS_S3_DATA_PREFIX = os.getenv("AWS_S3_DATA_PREFIX", "data")

# S3 paths
S3_DATA_PATH = f"s3://{AWS_S3_BUCKET}/{AWS_S3_DATA_PREFIX}"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model file names
MODEL_FILENAME = "nutrition_grade_model.pkl"
METADATA_FILENAME = "model_metadata.pkl"

# Model paths
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
METADATA_PATH = MODELS_DIR / METADATA_FILENAME

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Data file names
RAW_DATA_FILENAME = "data.parquet"
SAMPLE_DATA_FILENAME = "sample_data.parquet"
CLEAN_DATA_FILENAME = "clean_data.parquet"
FEATURES_FILENAME = "features.csv"
LABELS_FILENAME = "labels.csv"
PREDICTIONS_FILENAME = "nutrition_grade_predictions.csv"

# Data paths
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME
SAMPLE_DATA_PATH = RAW_DATA_DIR / SAMPLE_DATA_FILENAME
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / CLEAN_DATA_FILENAME
FEATURES_PATH = PROCESSED_DATA_DIR / FEATURES_FILENAME
LABELS_PATH = PROCESSED_DATA_DIR / LABELS_FILENAME
PREDICTIONS_PATH = PROCESSED_DATA_DIR / PREDICTIONS_FILENAME

# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================

# Default preprocessing settings
DEFAULT_PREPROCESSING_CONFIG = {
    'columns_to_drop': [
        'energy_100g', 'saturated-fat_100g', 'fiber_100g', 'main_category',
        'categories_tags', 'brands_tags', 'countries_tags', 'serving_size',
        'created_t', 'year', 'country'
    ],
    'target_column': 'nutrition_grade_fr',
    'invalid_grades': ['unknown', 'not-applicable'],
    'outlier_threshold': 100,
    'outlier_columns': [
        'fat_100g', 'carbohydrates_100g', 'proteins_100g', 
        'salt_100g', 'sugars_100g'
    ],
    'sample_fraction': None,  # Set to 0.1 for development
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

# =============================================================================
# MODELING CONFIGURATION
# =============================================================================

# XGBoost default parameters
DEFAULT_XGBOOST_PARAMS = {
    'objective': 'multi:softprob',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'mlogloss'
}

# Training configuration
TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'early_stopping_rounds': 10
}

# Hyperparameter tuning configuration
HYPERPARAMETER_TUNING_CONFIG = {
    'grid_search': {
        'param_grid': {
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'n_estimators': [50, 100, 150],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        },
        'cv_folds': 5
    },
    'random_search': {
        'n_trials': 20,
        'cv_folds': 5,
        'random_state': 42
    },
    'bayesian_search': {
        'n_trials': 30,
        'cv_folds': 5,
        'random_state': 42
    }
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure loguru with tqdm integration if available
try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_directories() -> None:
    """Create all necessary directories if they don't exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def get_s3_data_path(filename: str) -> str:
    """Get full S3 path for a data file."""
    return f"{S3_DATA_PATH}/{filename}"

def get_model_info() -> Dict[str, Any]:
    """Get information about the current model setup."""
    return {
        'model_path': str(MODEL_PATH),
        'metadata_path': str(METADATA_PATH),
        'model_exists': MODEL_PATH.exists(),
        'metadata_exists': METADATA_PATH.exists()
    }

# Create directories on import
ensure_directories()
