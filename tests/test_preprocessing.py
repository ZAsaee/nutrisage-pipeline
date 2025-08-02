# tests/test_preprocessing.py
import pandas as pd
from src.preprocessing.steps import (
    handle_missing_values, remove_outliers,
    compute_feature_engineering, encode_labels
)
from src.config import settings

# Build a synthetic DataFrame matching feature_columns + label


def make_df():
    data = {col: [1.0, None, 3.0] for col in settings.feature_columns}
    data[settings.label_column] = ['a', 'b', 'c']
    return pd.DataFrame(data)


def test_handle_missing_values():
    df = make_df()
    clean = handle_missing_values(df)
    # Only rows without any NA remain
    assert not clean.isnull().any().any()
    # At least one row should remain
    assert len(clean) >= 1


def test_remove_outliers():
    # Create DataFrame with one extreme outlier
    values = [1.0, 2.0, 1000.0]
    df = pd.DataFrame({col: values for col in settings.feature_columns})
    filtered = remove_outliers(df)
    # Outlier row removed
    assert len(filtered) == 2


def test_feature_engineering_and_label_encoding():
    df = make_df().dropna()
    df = compute_feature_engineering(df)
    # New engineered feature present
    assert 'fat_carb_ratio' in df.columns
    df = encode_labels(df)
    # Label column should now be numeric
    assert pd.api.types.is_integer_dtype(df[settings.label_column])
