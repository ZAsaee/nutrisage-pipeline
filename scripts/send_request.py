#!/usr/bin/env python3
from config import settings
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse, urlunparse

# Ensure src package is on the path
sys.path.append(str(Path(__file__).resolve().parent / "src"))


def create_session(retries: int = 5, backoff_factor: float = 1.0) -> requests.Session:
    """
    Create a requests Session with retry/backoff for transient errors.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def wait_for_endpoint(session: requests.Session, url: str, timeout: int = 5, max_wait: int = 300) -> bool:
    """
    Poll the endpoint's health until it responds or max_wait seconds pass.
    Treat any HTTP response (including 404) as indication the service is up.
    """
    print(
        f"Waiting up to {max_wait}s for endpoint to become ready at {url}...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = session.get(url, timeout=timeout)
            print(f"Health check returned status {r.status_code}")
            return True
        except requests.exceptions.ConnectTimeout:
            pass
        except Exception as e:
            print(f"Health check error: {e}")
        print('.', end='', flush=True)
        time.sleep(5)
    print("\nERROR: Endpoint not responding within timeout.")
    return False


def load_batches(parquet_path: str, batch_size: int):
    df = pd.read_parquet(parquet_path)
    records = df.to_dict("records")
    return [records[i: i + batch_size] for i in range(0, len(records), batch_size)]


def send_batch(session: requests.Session, endpoint: str, batch: list, timeout: int = 10):
    try:
        print(f"→ POSTing to {endpoint}")
        resp = session.post(endpoint, json=batch, timeout=timeout)
        resp.raise_for_status()
        preds = resp.json()
        print(
            f"[BATCH] Sent {len(batch)} records, received {len(preds)} preds: {preds[:3]}...")
        return resp.status_code, preds
    except requests.exceptions.ConnectTimeout:
        print(f"[TIMEOUT] Connection timed out after {timeout}s")
        return None, []
    except Exception as e:
        print(f"[ERROR] {e}")
        return None, []


def main():
    parser = argparse.ArgumentParser(
        description="Load-test the /batch_predict endpoint with retries & optional health check."
    )
    parser.add_argument("--parquet-file", required=True,
                        help="Path to your Parquet data file.")
    parser.add_argument("--endpoint", required=True,
                        help="Full URL of /batch_predict endpoint.")
    parser.add_argument("--health-path", default="/health",
                        help="Health check path on host root.")
    parser.add_argument("--skip-health", action="store_true",
                        help="Skip the initial health check.")
    parser.add_argument("--batch-size", type=int,
                        default=50, help="Records per request.")
    parser.add_argument("--max-workers", type=int, default=5,
                        help="Parallel requests per wave.")
    parser.add_argument("--interval", type=float,
                        default=1.0, help="Seconds between waves.")
    parser.add_argument("--duration", type=float,
                        default=1.0, help="Run time in hours.")
    args = parser.parse_args()

    # Validate protocol (http vs. https)
    parsed_ep = urlparse(args.endpoint)
    if parsed_ep.scheme not in ("http", "https"):
        print(
            f"[ERROR] Endpoint URL must start with http:// or https://; got '{parsed_ep.scheme}'")
        sys.exit(1)
    print(f"[INFO] Protocol detected: {parsed_ep.scheme.upper()}")

    session = create_session(retries=5, backoff_factor=1.0)

    if not args.skip_health:
        # Derive host root from endpoint for health check
        root = urlunparse((parsed_ep.scheme, parsed_ep.netloc, '', '', '', ''))
        health_url = root + args.health_path
        if not wait_for_endpoint(session, health_url):
            sys.exit(1)
    else:
        print("Skipping health check as requested.")

    batches = load_batches(args.parquet_file, args.batch_size)
    end_time = time.time() + args.duration * 3600
    wave = 0

    print(
        f"Starting load test: {len(batches)} batches, {args.max_workers} workers, interval={args.interval}s, duration={args.duration}h")

    while time.time() < end_time:
        wave += 1
        print(f"[Wave {wave}] Dispatching {len(batches)} batches...")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(
                send_batch, session, args.endpoint, b) for b in batches]
            results = [f.result() for f in as_completed(futures)]

        statuses, _ = zip(*results)
        success = sum(1 for s in statuses if s == 200)
        print(f"[Wave {wave}] {success}/{len(batches)} succeeded")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
