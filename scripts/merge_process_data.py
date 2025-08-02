
from preprocessing.steps import (
    handle_missing_values,
    remove_outliers,
    compute_feature_engineering
)
from config import settings
import sys
from pathlib import Path
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Ensure src/ is on the path to import settings and preprocessing
sys.path.append(str(Path(__file__).resolve().parent / "src"))


def process_file(path: Path) -> pd.DataFrame:
    """
    Read a single parquet file, select feature columns, and apply preprocessing steps.
    Returns a cleaned DataFrame.
    """
    df = pd.read_parquet(path, columns=settings.feature_columns)
    df = handle_missing_values(df)
    df = remove_outliers(df)
    df = compute_feature_engineering(df)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Merge all parquet partitions, preprocess, and write a single Parquet file."
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root directory of Parquet data (e.g., data/raw)"
    )
    parser.add_argument(
        "--output-path", required=True,
        help="Path for the output merged Parquet file (e.g., merged_output.parquet)"
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Number of threads to use for parallel file reading/preprocessing"
    )
    args = parser.parse_args()

    root = Path(args.data_dir)
    parquet_paths = list(root.glob("year*/country*/*.parquet"))
    if not parquet_paths:
        print(f"No parquet files found under {root}/year*/country*.")
        sys.exit(1)

    print(
        f"Found {len(parquet_paths)} files. Processing with {args.max_workers} threads...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        dfs = list(executor.map(process_file, parquet_paths))

    print(f"Concatenating {len(dfs)} DataFrames...")
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Merged DataFrame has {len(merged_df):,} rows.")

    print(f"Writing merged DataFrame to {args.output_path}...")
    merged_df.to_parquet(args.output_path, index=False)
    print("Merge and write complete.")


if __name__ == "__main__":
    main()
