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

# Global variables for startup/shutdown
model_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting NutriSage API...")
    
    try:
        global model_manager
        model_manager = ModelManager()
        
        if model_manager.is_ready():
            logger.success("Model loaded successfully!")
        else:
            logger.error("Failed to load model")
            
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down NutriSage API...")


# Create FastAPI application
app = FastAPI(
    title="NutriSage API",
    description="ML-powered nutrition grade prediction service",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["predictions"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "NutriSage API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/v1/predict",
            "health": "/api/v1/health",
            "docs": "/docs"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    try:
        if model_manager and model_manager.is_ready():
            return {"status": "healthy", "model": "loaded"}
        else:
            return {"status": "unhealthy", "model": "not_loaded"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


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