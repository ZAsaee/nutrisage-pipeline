# src/preprocessing/steps.py
import pandas as pd
from src.config import settings


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Only look at numeric columns for outlier detection
    numeric = df.select_dtypes(include='number')
    Q1 = numeric.quantile(0.25)
    Q3 = numeric.quantile(0.75)
    IQR = Q3 - Q1

    # Build a boolean mask of rows to *keep*
    mask = ~((numeric < (Q1 - 1.5 * IQR))
             | (numeric > (Q3 + 1.5 * IQR))).any(axis=1)

    # Apply that mask to the full DataFrame (so labels stay aligned)
    return df.loc[mask]


def compute_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['fat_carb_ratio'] = df['fat_100g'] / (df['carbohydrates_100g'] + 1e-6)
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    raw_col = settings.label_column
    grade_map = {g.lower(): i for i, g in enumerate(['A', 'B', 'C', 'D', 'E'])}
    df[raw_col] = df[raw_col].str.lower().map(grade_map)
    return df
