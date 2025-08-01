# src/config.py
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict as ConfigDict
from typing import List


class Settings(BaseSettings):
    # Paths
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    preprocessed_data_file: str = "data/processed/preprocessed.parquet"
    dataset_file: str = "data/processed/dataset.parquet"
    model_output_path: str = "models/xgb_model.joblib"
    feature_importance_output: str = "feature_importance.csv"

    # AWS/S3
    s3_bucket_name: str | None = None
    aws_region: str = "us-east-1"

    # Data selection
    feature_columns: List[str] = [
        "energy-kcal_100g", "fat_100g", "carbohydrates_100g",
        "sodium_100g", "fiber_100g", "proteins_100g", "sugars_100g"
    ]
    label_column: str = "nutrition_grade_fr"

    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    bayes_search_iterations: int = 32

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False
    )


# Instantiate a global settings object
settings = Settings()
