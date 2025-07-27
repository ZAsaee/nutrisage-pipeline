"""
Utility functions for the NutriSage API.
"""

import pandas as pd
from loguru import logger

from src.modeling.predict import load_model_and_metadata, preprocess_input_data, make_predictions
from src.config import MODEL_PATH, METADATA_PATH, ensure_directories
from .models import NutritionInput, PredictionResponse, NutritionGrade


class ModelManager:
    """Singleton class to manage model loading and caching."""
    _instance = None
    _model = None
    _metadata = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._load_model()
    
    def _load_model(self):
        """Load the trained model and metadata."""
        try:
            logger.info("Loading model...")
            ensure_directories()
            self._model, self._metadata = load_model_and_metadata(MODEL_PATH, METADATA_PATH)
            logger.success("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @property
    def model(self):
        """Get the loaded model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def metadata(self):
        """Get the model metadata."""
        if self._metadata is None:
            self._load_model()
        return self._metadata
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self._model is not None and self._metadata is not None


def predict_nutrition_grade(nutrition_data: NutritionInput) -> PredictionResponse:
    """Make prediction for nutrition data."""
    try:
        # Get model manager
        model_manager = ModelManager()
        
        if not model_manager.is_ready():
            raise RuntimeError("Model not loaded")
        
        # Convert input to DataFrame
        input_dict = nutrition_data.dict()
        df = pd.DataFrame([input_dict])
        
        # Preprocess input data
        X = preprocess_input_data(df, model_manager.metadata, None)
        
        # Make prediction
        results = make_predictions(
            model_manager.model, 
            X, 
            model_manager.metadata, 
            return_probabilities=True
        )
        
        # Extract results
        predicted_grade = results['predicted_labels'][0]
        probabilities = results['probabilities'][0]
        confidence = max(probabilities)
        
        # Create response
        return PredictionResponse(
            nutrition_grade=NutritionGrade(predicted_grade),
            confidence=confidence,
            probabilities=dict(zip(['a', 'b', 'c', 'd', 'e'], probabilities))
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def get_health_status() -> dict:
    """Get health status of the API service."""
    try:
        model_manager = ModelManager()
        return {
            "status": "healthy",
            "model_loaded": model_manager.is_ready()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "model_loaded": False
        } 