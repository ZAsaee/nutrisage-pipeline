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
    energy_kcal_100g: float = Field(..., ge=0,
                                    description="Energy content per 100g (kcal)")
    fat_100g: float = Field(..., ge=0, description="Fat content per 100g")
    carbohydrates_100g: float = Field(..., ge=0,
                                      description="Carbohydrates per 100g")
    sugars_100g: float = Field(..., ge=0, description="Sugars per 100g")
    proteins_100g: float = Field(..., ge=0, description="Proteins per 100g")
    sodium_100g: float = Field(..., ge=0,
                               description="Sodium content per 100g")

    # Derived features (optional, will be calculated if not provided)
    fat_carb_ratio: float = Field(
        None, ge=0, description="Ratio of fat to carbohydrates")
    protein_carb_ratio: float = Field(
        None, ge=0, description="Ratio of protein to carbohydrates")
    protein_fat_ratio: float = Field(
        None, ge=0, description="Ratio of protein to fat")
    total_macros: float = Field(
        None, ge=0, description="Total of fat, carbohydrates, and proteins")


class PredictionResponse(BaseModel):
    """Response model for nutrition grade prediction."""
    nutrition_grade: NutritionGrade = Field(...,
                                            description="Predicted nutrition grade")
    confidence: float = Field(..., ge=0, le=1,
                              description="Prediction confidence score")
    probabilities: Dict[str,
                        float] = Field(..., description="Probability for each grade")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(...,
                               description="Whether model is loaded and ready")
