# src/util.py
import pandas as pd
from src.config import settings


def sample_df(df: pd.DataFrame, fraction: float = None, random_state: int = None,) -> pd.DataFrame:
    """
    Randomly sample a fraction of the DataFrame for faster experiments.
    If no fraction provided, returns original DataFrame.
    """
    frac = fraction if fraction is not None else 1.0
    rs = random_state or settings.random_state
    return df.sample(frac=frac, random_state=rs, axis=0)
