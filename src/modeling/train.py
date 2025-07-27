import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import typer

from nutrisage.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def load_and_preprocess_data(features_path: Path, labels_path: Path) -> tuple:
    """Load and preprocess the nutrition data."""
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)
    
    logger.info(f"Loading labels from {labels_path}")
    labels_df = pd.read_csv(labels_path)
    
    # Merge features and labels
    data = features_df.merge(labels_df, left_index=True, right_index=True, how='inner')
    
    # Handle missing values
    data = data.dropna()
    
    # Separate features and target
    # Assuming the target column is 'nutrition_grade' or 'grade'
    target_col = None
    for col in ['nutrition_grade', 'grade', 'nutrition_grade_fr', 'nutrition-score-fr_100g']:
        if col in data.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError("Could not find nutrition grade column. Available columns: " + str(data.columns.tolist()))
    
    # Remove non-numeric columns and target column
    feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    
    X = data[feature_cols]
    y = data[target_col]
    
    # Encode target labels if they're strings
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        logger.info(f"Encoded labels: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    else:
        label_encoder = None
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Feature columns: {feature_cols}")
    
    return X, y, feature_cols, label_encoder


def train_xgboost_model(X: pd.DataFrame, y: pd.Series, feature_cols: list) -> tuple:
    """Train XGBoost model with hyperparameter tuning."""
    logger.info("Training XGBoost model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # XGBoost parameters for nutrition grade classification
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    
    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
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
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "nutrition_grade_model.pkl",
    metadata_path: Path = MODELS_DIR / "model_metadata.pkl",
):
    """Train XGBoost model for nutrition grade prediction."""
    logger.info("Starting nutrition grade prediction model training...")
    
    try:
        # Load and preprocess data
        X, y, feature_cols, label_encoder = load_and_preprocess_data(features_path, labels_path)
        
        # Train model
        model, X_test, y_test, y_pred, feature_importance = train_xgboost_model(X, y, feature_cols)
        
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
