#!/usr/bin/env python3
from config import settings
import sys
from pathlib import Path
import argparse
import pandas as pd
import requests
from pprint import pprint

# Ensure src on path for settings
sys.path.append(str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Test the /batch_predict endpoint with a sample from the clean Parquet and print the response."
    )
    parser.add_argument("--parquet-file", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    # Load and preview sample
    df = pd.read_parquet(args.parquet_file)
    sample = df.head(args.batch_size)
    records = sample.to_dict("records")

    print(f"Sending {len(records)} records to {args.endpoint}")
    pprint(records)  # helpful for debugging

    try:
        resp = requests.post(
            args.endpoint,
            json=records,  # send raw list — required by FastAPI
            timeout=10
        )
        print("HTTP status:", resp.status_code)
        print("Response body:", resp.text)
        resp.raise_for_status()
        print("Parsed JSON response:")
        pprint(resp.json())
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
