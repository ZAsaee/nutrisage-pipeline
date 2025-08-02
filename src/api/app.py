# src/api/app.py
from fastapi import FastAPI
from src.api.predict import predict_single, predict_batch
from src.logger import logger

logger.info("Starting FastAPI application")
app = FastAPI(title="NutriSage API")
<<<<<<< HEAD

# Add a health endpoint


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}

# (Optional) also handle root so GET / doesn’t 404


@app.get("/", include_in_schema=False)
def root():
    return {"message": "NutriSage inference up and running"}


@app.post("/predict")
async def predict_endpoint(item: dict):
    return predict_single(item)


=======


@app.post("/predict")
async def predict_endpoint(item: dict):
    return predict_single(item)


>>>>>>> origin/main
@app.post("/batch_predict")
async def batch_predict_endpoint(items: list[dict]):
    return predict_batch(items)
