#!/usr/bin/env python
"""
Simple test to check API connectivity.
"""

import os
import httpx
from loguru import logger


def test_api():
    """Test basic API connectivity."""

    # Get the target URL from environment
    target_url = os.getenv("API_URL", "http://localhost:8000")
    logger.info(f"Target URL: {target_url}")

    # Construct prediction URL
    api_url = target_url.rstrip("/") + "/api/predict"
    logger.info(f"Testing API at: {api_url}")

    # Sample data for prediction
    sample_data = {
        "energy_kcal_100g": 250.0,
        "fat_100g": 12.0,
        "carbohydrates_100g": 30.0,
        "sugars_100g": 15.0,
        "proteins_100g": 8.0,
        "sodium_100g": 0.5,
        "fat_carb_ratio": 0.4,
        "protein_carb_ratio": 0.267,
        "protein_fat_ratio": 0.667,
        "total_macros": 50.0
    }

    try:
        # Test with httpx
        with httpx.Client(timeout=10.0) as client:
            response = client.post(api_url, json=sample_data)
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response body: {response.text}")

            if response.status_code == 200:
                logger.success("API test successful!")
            else:
                logger.error(
                    f"API test failed with status {response.status_code}")

    except httpx.ConnectError as e:
        logger.error(f"Connection error: {e}")
        logger.error(
            "This usually means the API server is not running or not accessible")
    except Exception as e:
        logger.error(f"API test failed with exception: {e}")
        logger.error(f"Exception type: {type(e)}")


if __name__ == "__main__":
    test_api()
