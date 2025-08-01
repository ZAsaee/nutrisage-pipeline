# tests/test_data_utils.py
import pandas as pd
from src.data.utils import sample_df


def test_sample_df_full():
    df = pd.DataFrame({'a': range(10)})
    sampled = sample_df(df, fraction=1.0, random_state=0)
    assert set(sampled['a']) == set(df['a'])


def test_sample_df_half():
    df = pd.DataFrame({'a': range(10)})
    sampled = sample_df(df, fraction=0.5, random_state=0)
    assert len(sampled) == 5
