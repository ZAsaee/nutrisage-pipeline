#!/usr/bin/env python
"""run_inference.py

Send rows from S3 files through the NutriSage `/predict` API in parallel.

Usage (environment variables):
  TARGET_URL    – Base URL of the running NutriSage API (e.g. https://my-alb.amazonaws.com)
  INPUT_PREFIX  – S3 URI prefix containing CSV/Parquet files (e.g. s3://nutrisage/data)
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
from loguru import logger

# ---------------------------------------------------------------------------
# Configuration via env vars -------------------------------------------------
# ---------------------------------------------------------------------------
API_ROUTE = "/predict"
TARGET_URL = os.getenv("TARGET_URL")
INPUT_PREFIX = os.getenv("INPUT_PREFIX", "s3://nutrisage/data")
WORKERS = int(os.getenv("WORKERS", "50"))
TIMEOUT = float(os.getenv("TIMEOUT", "10"))  # seconds per request

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
    """Load a small CSV or Parquet file from S3 into a DataFrame."""
    logger.debug(f"Downloading s3://{bucket}/{key}")
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()

    if key.lower().endswith(".parquet"):
        from io import BytesIO

        return pd.read_parquet(BytesIO(body))
    elif key.lower().endswith((".csv", ".csv.gz")):
        import io

        return pd.read_csv(io.BytesIO(body))
    else:
        raise ValueError(f"Unsupported file type for {key}")

# ---------------------------------------------------------------------------
# Async HTTP logic -----------------------------------------------------------
# ---------------------------------------------------------------------------


def row_to_payload(row: pd.Series) -> Dict[str, float]:
    mapping = {
        "energy-kcal_100g": "energy_kcal_100g",
        "fat_100g": "fat_100g",
        "carbohydrates_100g": "carbohydrates_100g",
        "sugars_100g": "sugars_100g",
        "proteins_100g": "proteins_100g",
        "sodium_100g": "sodium_100g",
    }
    payload = {api_name: float(row[model_name])
               for model_name, api_name in mapping.items()}
    return payload


async def _post_row(client: httpx.AsyncClient, payload: Dict[str, float]):
    start = time.perf_counter()
    r = await client.post(TARGET_URL, json=payload)
    latency = (time.perf_counter() - start) * 1000
    r.raise_for_status()
    logger.debug(f"200 OK  {latency:6.1f} ms")


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
        payloads = [row_to_payload(row) for _, row in df.iterrows()]
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


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
