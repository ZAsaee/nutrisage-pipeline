# src/preprocessing/pipeline.py
from src.logger import logger
from src.config import settings
from src.data.storage import load_parquet
from src.preprocessing.steps import (
    handle_missing_values,
    remove_outliers,
    compute_feature_engineering,
    encode_labels,
)
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset to model-ready format.")
    parser.add_argument("--input", default=settings.dataset_file,
                        help="Path to sampled/parquet dataset")
    parser.add_argument("--output", default=settings.preprocessed_data_file,
                        help="Path to save preprocessed dataset")
    args = parser.parse_args()

    logger.info("Loading data for preprocessing from %s", args.input)
    df = load_parquet(args.input)
    logger.info("Loaded %d rows", len(df))

    cols = settings.feature_columns + [settings.label_column]
    df = df[cols]
    logger.info("Filtered to cols: %s", cols)

    for fn in (handle_missing_values, remove_outliers, compute_feature_engineering, encode_labels):
        logger.info("Applying %s", fn.__name__)
        df = fn(df)
        logger.info("Rows after %s: %d", fn.__name__, len(df))

    logger.info("Saving preprocessed data to %s", args.output)
    df.to_parquet(args.output, index=False)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
