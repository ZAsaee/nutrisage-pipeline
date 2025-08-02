#!/usr/bin/env python3
from config import settings
import sys
import time
from pathlib import Path
import argparse
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor

# Ensure src/ is on the path to import settings if needed
sys.path.append(str(Path(__file__).resolve().parent / "src"))


def load_batches(parquet_path: Path, batch_size: int):
    """
    Load the single clean Parquet file into memory, split into record batches.
    """
    df = pd.read_parquet(parquet_path, columns=settings.feature_columns)
    records = df.to_dict("records")
    return [records[i: i + batch_size] for i in range(0, len(records), batch_size)]


def timed_load_loop(parquet_path: Path, endpoint: str, batch_size: int, max_workers: int, duration_hours: float):
    """
    Continuously send the DataFrame batches to the endpoint for the specified duration.
    """
    # Prepare HTTP session with connection pooling
    session = requests.Session()
    adapter = HTTPAdapter(pool_maxsize=max_workers * 2)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Pre-load batches
    print(f"Loading and batching data from {parquet_path}...")
    batches = load_batches(parquet_path, batch_size)
    total_batches = len(batches)
    print(f"Prepared {total_batches} batches (batch_size={batch_size}).")

    # Set up loop
    end_time = time.time() + duration_hours * 3600
    cycle = 0
    print(f"Starting load loop for {duration_hours} hours...")

    def send_batch(batch):
        resp = session.post(endpoint, json={"items": batch})
        resp.raise_for_status()

    while time.time() < end_time:
        cycle += 1
        print(f"Cycle {cycle}: sending {total_batches} batches...")
        start_cycle = time.time()
        # Send all batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(send_batch, batches)
        cycle_time = time.time() - start_cycle
        print(f"Cycle {cycle} completed in {cycle_time:.2f}s")

    total_time = duration_hours * 3600
    print(
        f"Timed load loop finished after {total_time:.0f}s ({duration_hours} hours) over {cycle} cycles.")


def main():
    parser = argparse.ArgumentParser(
        description="Loop sending a single clean Parquet file in batches to trigger autoscaling for a set duration."
    )
    parser.add_argument(
        "--parquet-file", required=True,
        help="Path to the clean Parquet file"
    )
    parser.add_argument(
        "--endpoint", required=True,
        help="Full URL of the /batch_predict endpoint"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="Number of records per request"
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Parallel workers for sending batches"
    )
    parser.add_argument(
        "--duration-hours", type=float, default=3.0,
        help="Total run time in hours"
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet_file)
    timed_load_loop(
        parquet_path,
        args.endpoint,
        args.batch_size,
        args.max_workers,
        args.duration_hours,
    )


if __name__ == "__main__":
    main()
