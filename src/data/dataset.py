# src/dataset.py
import argparse
from src.data.storage import load_parquet
from src.data.utils import sample_df
from src.config import settings
from src.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Load and sample raw dataset.")
    parser.add_argument("--input", default=settings.processed_data_path,
                        help="Path to raw parquet (local or s3://)")
    parser.add_argument("--sample_fraction", type=float, default=None,
                        help="Fraction to sample for quick experiments")
    parser.add_argument("--output", default=settings.dataset_file,
                        help="Path to save sampled dataset")
    args = parser.parse_args()

    logger.info("Loading raw data from %s", args.input)
    df = load_parquet(args.input)
    logger.info("Loaded %d rows", len(df))

    if args.sample_fraction:
        logger.info("Sampling fraction=%s", args.sample_fraction)
        df = sample_df(df, args.sample_fraction, settings.random_state)
        logger.info("Sampled down to %d rows", len(df))

    logger.info("Saving dataset to %s", args.output)
    df.to_parquet(args.output, index=False)
    logger.info("Done")


if __name__ == "__main__":
    main()
