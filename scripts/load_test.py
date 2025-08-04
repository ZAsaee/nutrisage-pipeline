import time
import random
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

# Data file path
PARQUET_FILE = "data/processed/full_data.parquet"

# Inference endpoint URL
ENDPOINT = "http://nutris-Publi-FzH9ZcYsvl8x-2110404969.us-east-1.elb.amazonaws.com/batch_predict"

# Number of records per request
BATCH_SIZE = 100

# Define traffic waves (duration in seconds, requests per second)
SCHEDULE = [
    (180, 5),  # 3 min @ 10 RPS / below both CPU & RPS thresholds-scale-in
    (180, 50),  # 3 min @ 50 RPS / above RPS but maybe below CPU-moderate
    (300, 200),  # 5 min @ 200 RPS / well above thresholds-scale-out-Heavy
    (300, 350),  # 5 min @ 350 RPS / well above thresholds-scale-out-Extra Heavy
    (180, 5),
    (180, 50)
]

df = pd.read_parquet(PARQUET_FILE)
records = df.to_dict("records")
batches = [records[i: i+BATCH_SIZE]
           for i in range(0, len(records), BATCH_SIZE)]

# Shared HTTP session
session = requests.Session()


def send_batch(batch):
    resp = session.post(ENDPOINT, json=batch)
    return resp.status_code


if __name__ == "__main__":
    start = time.time()
    DURATION = 5 * 24 * 60  # Run for five cycles of defined schedule
    max_reps = max(rps for _, rps in SCHEDULE)
    pool = ThreadPoolExecutor(max_workers=max_reps)
    try:
        while time.time() - start < DURATION:
            for duration, rps in SCHEDULE:
                interval = 1.0 / rps
                print(f"Starting wave: {duration}s @ {rps} RPS")
                deadline = time.time() + duration
                while time.time() < deadline:
                    batch = random.choice(batches)
                    pool.submit(send_batch, batch)
                    time.sleep(interval)
    finally:
        pool.shutdown(wait=True)
        print("Load test completed.")
