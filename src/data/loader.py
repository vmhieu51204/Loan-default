"""
Data loader module for reading raw loan dataset files.
"""

import os
from typing import Optional
import pandas as pd


def load_raw_data(
    filepath: str,
    compression: Optional[str] = "gzip",
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load raw loan dataset from disk.

    Args:
        filepath: Path to the dataset file (.csv or .csv.gz / .gz).
        compression: Compression format (e.g., 'gzip', 'infer', or None).
        sample_size: Optional number of rows to sample for fast testing.
        random_state: Seed for reproducible random sampling.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    print(f"Loading raw data from: {filepath} ...")
    # If sample_size is requested, read only an initial batch to avoid decompressing gigabytes
    nrows = (sample_size * 3) if sample_size is not None else None
    df = pd.read_csv(filepath, compression=compression, low_memory=False, nrows=nrows)
    print(f"Loaded raw dataset with shape: {df.shape}")

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"Sampled {sample_size} records. New shape: {df.shape}")

    return df
