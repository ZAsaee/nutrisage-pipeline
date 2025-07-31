#!/usr/bin/env python
"""run_inference.py

Send rows from S3 files through the NutriSage `/predict` API in parallel.

Usage (environment variables):
  TARGET_URL    – Base URL of the running NutriSage API (e.g. https://my-alb.amazonaws.com)
  INPUT_PREFIX  – S3 URI prefix containing Parquet files (e.g. s3://nutrisage/data)
  WORKERS       – Concurrency; number of parallel requests (default: 50)

Example:
  TARGET_URL=https://alb-xyz.elb.amazonaws.com \
  INPUT_PREFIX=s3://nutrisage/data \
  WORKERS=40 python scripts/run_inference.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import boto3
import httpx
import pandas as pd
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Configuration via env vars -------------------------------------------------
# ---------------------------------------------------------------------------
API_ROUTE = "/api/v1/predict"
TARGET_URL = os.getenv("TARGET_URL")
INPUT_PREFIX = os.getenv("INPUT_PREFIX", "s3://nutrisage/data")
WORKERS = int(os.getenv("WORKERS", "50"))
TIMEOUT = float(os.getenv("TIMEOUT", "10"))  # seconds per request
DEBUG = os.getenv("DEBUG", "false").lower() == "true"  # debug mode

if not TARGET_URL:
    logger.error("Set TARGET_URL env var (e.g. https://alb/)")
    sys.exit(1)

# strip trailing slash to avoid double //
TARGET_URL = TARGET_URL.rstrip("/") + API_ROUTE

# ---------------------------------------------------------------------------
# S3 helper functions --------------------------------------------------------
# ---------------------------------------------------------------------------

session = boto3.Session()
s3_client = session.client("s3")


def _split_s3_uri(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError("INPUT_PREFIX must begin with s3://")
    bucket, key = uri.replace("s3://", "", 1).split("/", 1)
    return bucket, key


def list_s3_objects(prefix: str):
    bucket, key_prefix = _split_s3_uri(prefix)
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            yield bucket, obj["Key"]


def load_dataframe(bucket: str, key: str) -> pd.DataFrame:
    """Load a small Parquet file from S3 into a DataFrame."""
    logger.debug(f"Downloading s3://{bucket}/{key}")
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()

    if key.lower().endswith(".parquet"):
        from io import BytesIO

        # Load only the required columns
        required_columns = [
            "energy-kcal_100g",
            "fat_100g",
            "carbohydrates_100g",
            "sugars_100g",
            "proteins_100g",
            "sodium_100g"
        ]

        df = pd.read_parquet(BytesIO(body), columns=required_columns)

        # Validate data quality
        logger.debug(f"Loaded DataFrame shape: {df.shape}")

        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Missing values found: {missing_counts.to_dict()}")

        # Check for infinite values
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        if inf_counts.sum() > 0:
            logger.warning(f"Infinite values found: {inf_counts.to_dict()}")

        # Check for negative values (which shouldn't exist for nutrition data)
        negative_counts = (df.select_dtypes(include=[np.number]) < 0).sum()
        if negative_counts.sum() > 0:
            logger.warning(
                f"Negative values found: {negative_counts.to_dict()}")

        return df
    elif key.lower().endswith((".csv", ".csv.gz")):
        import io

        return pd.read_csv(io.BytesIO(body))
    else:
        raise ValueError(f"Unsupported file type for {key}")


def row_to_payload(row: pd.Series) -> Dict[str, float]:
    """
    Convert a DataFrame row to API payload with derived features.

    Creates the 4 derived features that the model expects:
    - fat_carb_ratio: fat/carbohydrates
    - protein_carb_ratio: protein/carbohydrates  
    - protein_fat_ratio: protein/fat
    - total_macros: fat + carbohydrates + proteins
    """
    # Map basic nutrition columns from model names to API names
    mapping = {
        "energy-kcal_100g": "energy_kcal_100g",
        "fat_100g": "fat_100g",
        "carbohydrates_100g": "carbohydrates_100g",
        "sugars_100g": "sugars_100g",
        "proteins_100g": "proteins_100g",
        "sodium_100g": "sodium_100g",
    }

    # Create payload with basic features, handling null values
    payload = {}
    for model_name, api_name in mapping.items():
        value = row[model_name]
        # Handle null values by replacing with 0.0
        if pd.isna(value):
            payload[api_name] = 0.0
        else:
            payload[api_name] = float(value)

    # Add derived features that the model expects with proper division handling
    # These match what the API creates in utils.py

    # fat_carb_ratio: fat/carbohydrates
    if payload['carbohydrates_100g'] > 0:
        payload['fat_carb_ratio'] = payload['fat_100g'] / \
            payload['carbohydrates_100g']
    else:
        payload['fat_carb_ratio'] = 0.0

    # protein_carb_ratio: protein/carbohydrates
    if payload['carbohydrates_100g'] > 0:
        payload['protein_carb_ratio'] = payload['proteins_100g'] / \
            payload['carbohydrates_100g']
    else:
        payload['protein_carb_ratio'] = 0.0

    # protein_fat_ratio: protein/fat
    if payload['fat_100g'] > 0:
        payload['protein_fat_ratio'] = payload['proteins_100g'] / \
            payload['fat_100g']
    else:
        payload['protein_fat_ratio'] = 0.0

    # total_macros: fat + carbohydrates + proteins
    payload['total_macros'] = payload['fat_100g'] + \
        payload['carbohydrates_100g'] + payload['proteins_100g']

    # Cap extreme values to prevent API rejection
    for key, value in payload.items():
        if not np.isfinite(value):
            payload[key] = 0.0
        elif abs(value) > 1000:  # Cap very large values
            payload[key] = 1000.0 if value > 0 else -1000.0

    return payload


async def _post_row(client: httpx.AsyncClient, payload: Dict[str, float]):
    start = time.perf_counter()
    try:
        r = await client.post(TARGET_URL, json=payload)
        latency = (time.perf_counter() - start) * 1000
        r.raise_for_status()
        logger.debug(f"200 OK  {latency:6.1f} ms")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP {e.response.status_code}: {e.response.text}")
        raise
    except httpx.ConnectError as e:
        logger.error(f"Connection error: {e}")
        raise
    except Exception as e:
        logger.error(f"Request error: {e}")
        raise


async def worker(name: str, queue: asyncio.Queue):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        while True:
            payloads = await queue.get()
            if payloads is None:
                queue.task_done()
                break
            for payload in payloads:
                try:
                    await _post_row(client, payload)
                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    logger.error(f"Payload: {payload}")
            queue.task_done()


async def main():
    queue: asyncio.Queue = asyncio.Queue(maxsize=WORKERS * 2)

    # start workers
    tasks: List[asyncio.Task] = []
    for i in range(WORKERS):
        tasks.append(asyncio.create_task(worker(f"w{i}", queue)))

    # enqueue data
    for bucket, key in list_s3_objects(INPUT_PREFIX):
        df = load_dataframe(bucket, key)
        logger.info(f"Processing {len(df)} samples from s3://{bucket}/{key}")
        logger.info(f"Creating derived features for model compatibility...")

        payloads = [row_to_payload(row) for _, row in df.iterrows()]

        # Debug: show first few payloads
        if DEBUG and len(payloads) > 0:
            logger.info(f"First payload example: {payloads[0]}")
            if len(payloads) > 1:
                logger.info(f"Second payload example: {payloads[1]}")

        # break into chunks of 100 requests to keep memory down
        chunk_size = 100
        for i in range(0, len(payloads), chunk_size):
            await queue.put(payloads[i: i + chunk_size])

    # wait until all tasks finished
    await queue.join()

    # stop workers
    for _ in tasks:
        await queue.put(None)
    await asyncio.gather(*tasks)

    logger.success("Inference completed successfully!")
    logger.info(
        f"Processed data with derived features (fat_carb_ratio, protein_carb_ratio, protein_fat_ratio, total_macros)")
    logger.info(f"All predictions sent to {TARGET_URL}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
