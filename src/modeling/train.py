import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def run_preprocessing_pipeline(raw_data_path: Path, sample_fraction: float = None) -> tuple:
    """Run the complete preprocessing pipeline."""
    from src.preprocessing import NutritionDataPreprocessor
    
    logger.info("Running preprocessing pipeline...")
    
    # Initialize preprocessor with sampling if specified
    config = {}
    if sample_fraction:
        config['sample_fraction'] = sample_fraction
        config['random_state'] = 42
    
    preprocessor = NutritionDataPreprocessor(config)
    
    # Run preprocessing
    features_df, target_df, feature_columns, target_col = preprocessor.preprocess(
        raw_data_path, save_intermediate=True
    )
    
    # Prepare for modeling
    X = features_df
    y = target_df[target_col]
    
    # Encode target labels if they're strings
    if y.dtype == 'string':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        logger.info(f"Encoded labels: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    else:
        label_encoder = None
    
    logger.info(f"Preprocessing completed. Final shapes:")
    logger.info(f"  Features: {X.shape}")
    logger.info(f"  Target: {y.shape}")
    logger.info(f"  Feature columns: {len(feature_columns)}")
    
    return X, y, feature_columns, label_encoder


def perform_hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series, n_classes: int) -> dict:
    """Perform hyperparameter tuning using GridSearchCV."""
    logger.info("Starting hyperparameter tuning...")
    
    # Base parameters
    base_params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    
    # Define parameter grid for tuning (reduced for faster execution)
    param_grid = {
        'max_depth': [4, 6],
        'learning_rate': [0.1, 0.15],
        'n_estimators': [100],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    # Create base model
    base_model = xgb.XGBClassifier(**base_params)
    
    # Perform grid search
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    return grid_search.best_params_


def train_xgboost_model(X: pd.DataFrame, y: pd.Series, feature_cols: list, tune_hyperparameters: bool = False) -> tuple:
    """Train XGBoost model with optional hyperparameter tuning."""
    logger.info("Training XGBoost model...")
    
    # Split data (y should already be encoded from preprocessing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # XGBoost parameters for nutrition grade classification
    base_params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    
    if tune_hyperparameters:
        logger.info("Running hyperparameter tuning...")
        best_params = perform_hyperparameter_tuning(X_train, y_train, len(np.unique(y)))
        params = {**base_params, **best_params}
        logger.info(f"Using tuned parameters: {best_params}")
    else:
        # Use default parameters
        params = {
            **base_params,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
     
    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 most important features:")
    for _, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, X_test, y_test, y_pred, feature_importance


def save_model_and_metadata(
    model: xgb.XGBClassifier,
    feature_cols: list,
    label_encoder: LabelEncoder,
    feature_importance: pd.DataFrame,
    model_path: Path,
    metadata_path: Path
) -> None:
    """Save the trained model and metadata."""
    # Create models directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    logger.info(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        'feature_columns': feature_cols,
        'label_encoder': label_encoder,
        'feature_importance': feature_importance.to_dict('records'),
        'model_type': 'xgboost',
        'num_classes': len(model.classes_),
        'classes': model.classes_.tolist()
    }
    
    logger.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)


@app.command()
def main(
    raw_data_path: Path = RAW_DATA_DIR / "sample_data.parquet",
    model_path: Path = MODELS_DIR / "nutrition_grade_model.pkl",
    metadata_path: Path = MODELS_DIR / "model_metadata.pkl",
    sample_fraction: float = None,
    tune_hyperparameters: bool = typer.Option(False, "--tune", help="Run hyperparameter tuning"),
):
    """Train XGBoost model for nutrition grade prediction."""
    logger.info("Starting nutrition grade prediction model training...")
    
    try:
        # Check if raw data exists
        if not raw_data_path.exists():
            raise FileNotFoundError(f"Raw data not found at {raw_data_path}")
        
        # Run preprocessing pipeline (this handles all data preparation)
        logger.info(f"Loading and preprocessing data from {raw_data_path}")
        X, y, feature_cols, label_encoder = run_preprocessing_pipeline(raw_data_path, sample_fraction)
        
        # Train model
        model, X_test, y_test, y_pred, feature_importance = train_xgboost_model(
            X, y, feature_cols, tune_hyperparameters
        )
        
        # Save model and metadata
        save_model_and_metadata(
            model, feature_cols, label_encoder, feature_importance, model_path, metadata_path
        )
        
        # Print classification report
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        logger.success("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    app()
