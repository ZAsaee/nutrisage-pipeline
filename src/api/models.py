"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict
from enum import Enum


class NutritionGrade(str, Enum):
    """Valid nutrition grades."""
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    E = "e"


class NutritionInput(BaseModel):
    """Input model for nutrition data prediction."""
    fat_100g: float = Field(..., ge=0, description="Fat content per 100g")
    carbohydrates_100g: float = Field(..., ge=0, description="Carbohydrates per 100g")
    proteins_100g: float = Field(..., ge=0, description="Proteins per 100g")
    salt_100g: float = Field(..., ge=0, description="Salt content per 100g")
    sugars_100g: float = Field(..., ge=0, description="Sugars per 100g")
    fiber_100g: float = Field(..., ge=0, description="Fiber per 100g")
    saturated_fat_100g: float = Field(..., ge=0, description="Saturated fat per 100g")


class PredictionResponse(BaseModel):
    """Response model for nutrition grade prediction."""
    nutrition_grade: NutritionGrade = Field(..., description="Predicted nutrition grade")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    probabilities: Dict[str, float] = Field(..., description="Probability for each grade")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded and ready") 