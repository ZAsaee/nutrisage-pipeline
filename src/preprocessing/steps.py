# src/preprocessing/steps.py
import pandas as pd
from src.config import settings


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


def compute_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['fat_carb_ratio'] = df['fat_100g'] / (df['carbohydrates_100g'] + 1e-6)
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df[settings.label_column] = df['nutrition_grade'].map(
        {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4})
    return df
