"""
API endpoints for the NutriSage nutrition grade prediction service.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from .models import NutritionInput, PredictionResponse, HealthResponse
from .utils import predict_nutrition_grade, get_health_status

# Create router
router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, summary="Predict nutrition grade")
async def predict_nutrition_grade_endpoint(nutrition_data: NutritionInput):
    """Predict nutrition grade for a food product."""
    try:
        prediction = predict_nutrition_grade(nutrition_data)
        logger.info(f"Prediction: {prediction.nutrition_grade} (confidence: {prediction.confidence:.3f})")
        return prediction
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """Check the health status of the API service."""
    try:
        health_data = get_health_status()
        return HealthResponse(**health_data)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(status="unhealthy", model_loaded=False) 