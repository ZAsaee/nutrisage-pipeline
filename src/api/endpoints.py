"""
API endpoints for the NutriSage nutrition grade prediction service.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import time

# CloudWatch Embedded Metrics
try:
    from aws_embedded_metrics import metric_scope  # type: ignore

    @metric_scope
    def _record_inference_metrics(metrics, latency_ms: float, confidence: float, grade: str):
        """Emit inference latency & confidence to CloudWatch."""
        metrics.put_metric("ModelLatencyMs", latency_ms, "Milliseconds")
        metrics.put_metric("Confidence", confidence)
        metrics.set_property("grade", grade)

except ModuleNotFoundError:  # pragma: no cover
    def _record_inference_metrics(*args, **kwargs):  # type: ignore
        """Fallback no-op when aws_embedded_metrics isn't present."""
        return None

from .models import NutritionInput, PredictionResponse, HealthResponse
from .utils import predict_nutrition_grade, get_health_status

# Create router
router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, summary="Predict nutrition grade")
async def predict_nutrition_grade_endpoint(nutrition_data: NutritionInput):
    """Predict nutrition grade for a food product and emit CW metrics."""
    start_ts = time.perf_counter()
    try:
        prediction = predict_nutrition_grade(nutrition_data)

        # Emit latency / confidence metrics (no-op if library missing)
        latency_ms = (time.perf_counter() - start_ts) * 1000.0
        _record_inference_metrics(
            latency_ms, prediction.confidence, prediction.nutrition_grade.value)

        logger.info(
            f"Prediction: {prediction.nutrition_grade} (confidence: {prediction.confidence:.3f}, latency: {latency_ms:.1f} ms)"
        )
        return prediction
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """Check the health status of the API service."""
    try:
        health_data = get_health_status()
        if health_data["status"] == "warming_up":
            return JSONResponse(
                status_code=503,
                content=health_data,
            )
        return HealthResponse(**health_data)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(status="unhealthy", model_loaded=False)
