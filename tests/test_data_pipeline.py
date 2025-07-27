"""
Minimal tests for data pipeline using sample data.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.dataset import load_from_local, validate_nutrition_data


def test_load_sample_data():
    """Test loading sample nutrition data."""
    sample_path = Path(__file__).parent / "data" / "sample_nutrition_data.csv"
    df = load_from_local(sample_path)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert 'nutrition_grade_fr' in df.columns
    assert 'fat_100g' in df.columns


def test_validate_sample_data():
    """Test validation of sample nutrition data."""
    sample_path = Path(__file__).parent / "data" / "sample_nutrition_data.csv"
    df = load_from_local(sample_path)
    
    result = validate_nutrition_data(df)
    assert result['is_valid'] == True
    assert len(result['missing_columns']) == 0


def test_sample_data_distribution():
    """Test sample data has good grade distribution."""
    sample_path = Path(__file__).parent / "data" / "sample_nutrition_data.csv"
    df = load_from_local(sample_path)
    
    grades = df['nutrition_grade_fr'].value_counts()
    assert len(grades) >= 4  # At least 4 different grades (a, b, c, d, e)
    assert 'a' in grades.index
    assert 'b' in grades.index
    assert 'c' in grades.index
    assert 'd' in grades.index
    assert 'e' in grades.index 