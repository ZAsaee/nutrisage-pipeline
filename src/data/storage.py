# src/storage.py
import pandas as pd
import s3fs
from src.config import settings


def load_parquet(path: str) -> pd.DataFrame:
    """
    Load a parquet file from S3 or local into a DataFrame.
    """
    if path.startwith("s3://") or settings.s3_bucket_name and \
            path.startswith(settings.s3_bucket_name):
        fs = s3fs.S3FileSystem(region_name=settings.aws_region)
        return pd.read_parquet(path, filesystem=fs)
    return pd.read_parquet(path)
