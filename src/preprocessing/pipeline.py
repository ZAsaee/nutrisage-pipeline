# src/preprocessing/pipeline.py
import argparse
from src.data.storage import load_parquet
from src.preprocessing.steps import (
    handle_missing_values,
    remove_outliers,
    compute_feature_engineering,
    encode_labels,
)
from src.config import settings


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset to model-ready format.")
    parser.add_argument("--input", default=settings.dataset_file,
                        help="Path to sampled/parquet dataset")
    parser.add_argument("--output", default=settings.preprocessed_data_file,
                        help="Path to save preprocessed dataset")
    args = parser.parse_args()

    df = load_parquet(args.input)
    cols = settings.feature_columns + [settings.label_column]
    df = df[cols]
    df = handle_missing_values(df)
    df = remove_outliers(df)
    df = compute_feature_engineering(df)
    df = encode_labels(df)
    df.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
