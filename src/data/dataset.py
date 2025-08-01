# src/dataset.py
import argparse
from src.data.storage import load_parquet
from src.data.utils import sample_df
from src.config import settings


def main():
    parser = argparse.ArgumentParser(
        description="Load and sample raw dataset.")
    parser.add_argument("--input", default=settings.dataset_file,
                        help="Path to raw parquet (local or s3://)")
    parser.add_argument("--sample-fraction", type=float, default=None,
                        help="Fraction to sample for quick experiments")
    parser.add_argument("--output", default=settings.dataset_file,
                        help="Path to save sampled dataset")
    args = parser.parse_args()

    df = load_parquet(args.input)
    if args.sample_fraction:
        df = sample_df(df, args.sample_fraction)
    df.to_parquet(args.output, index=False)

    if __name__ == "__main__":
        main()
