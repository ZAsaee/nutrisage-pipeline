# src/storage.py
import pandas as pd
import s3fs
# import pyarrow.dataset as ds
import glob
from src.config import settings
import argparse


def load_parquet(path: str) -> pd.DataFrame:
    """
    Load a Parquet file or partitioned dataset (directory of .parquet files)
    from local or S3 into a DataFrame.
    """
    if path.startswith("s3://") or settings.s3_bucket_name and \
            path.startswith(settings.s3_bucket_name):
        fs = s3fs.S3FileSystem(region_name=settings.aws_region)
        return pd.read_parquet(path, filesystem=fs)
    else:
        files = glob.glob(f"{path}/*.parquet")
        if not files:
            raise FileNotFoundError(
                f"No .parquet files found in directory: {path}")
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)
