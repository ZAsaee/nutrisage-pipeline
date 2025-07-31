"""
FastAPI application for the NutriSage nutrition grade prediction service.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger
import uvicorn

from .endpoints import router
from .utils import ModelManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model manager and other resources on startup."""
    logger.info("Starting NutriSage API...")
    
    # Initialize model manager
    try:
        ModelManager()
        logger.success("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")

    yield
    
    logger.info("Shutting down NutriSage API...")


# Create app
app = FastAPI(
    title="NutriSage API",
    description="API for predicting nutrition grades of food products.",
    version="1.0.0",
    lifespan=lifespan
)

# Include router
app.include_router(router, prefix="/api")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "NutriSage API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/predict",
            "health": "/api/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        log_level="info"
    ) 