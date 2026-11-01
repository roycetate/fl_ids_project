"""
clean_data.py
-------------
Cleans the raw, concatenated CICIDS2017 DataFrame:
  - drops identifier / leakage-prone columns (IPs, ports, timestamps, etc.)
  - replaces +-Inf (common in 'Flow Bytes/s' and 'Flow Packets/s') with NaN
  - drops or imputes NaN rows
  - drops exact duplicate rows
  - coerces feature columns to numeric, dropping anything that fails
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def drop_irrelevant_columns(df: pd.DataFrame, drop_columns: List[str]) -> pd.DataFrame:
    present = [c for c in drop_columns if c in df.columns]
    if present:
        logger.info("Dropping %d identifier/leakage column(s): %s", len(present), present)
        df = df.drop(columns=present)
    if "__source_file__" in df.columns:
        df = df.drop(columns=["__source_file__"])
    return df


def replace_infinities(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(df[numeric_cols]).sum().sum()
    if n_inf:
        logger.info("Replacing %d infinite value(s) with NaN", int(n_inf))
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def coerce_numeric_features(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Force every non-label column to numeric; anything that can't convert becomes NaN."""
    feature_cols = [c for c in df.columns if c != label_column]
    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def drop_missing(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    before = len(df)
    # Never allow missing labels through.
    df = df.dropna(subset=[label_column])
    # For features, drop rows with any remaining NaN (CICIDS2017 NaNs are a
    # small fraction of rows, so dropping is safe and avoids biasing an
    # imputation strategy into a security-relevant feature).
    df = df.dropna()
    after = len(df)
    if before != after:
        logger.info("Dropped %d row(s) with missing values (%.3f%% of data)",
                     before - after, 100 * (before - after) / before)
    return df


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        logger.info("Dropped %d duplicate row(s)", before - after)
    return df


def drop_constant_columns(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Drop feature columns with zero variance -- they carry no signal and
    can break scalers (division by zero std)."""
    feature_cols = [c for c in df.columns if c != label_column]
    nunique = df[feature_cols].nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        logger.info("Dropping %d constant column(s): %s", len(constant_cols), constant_cols)
        df = df.drop(columns=constant_cols)
    return df


def clean_dataset(df: pd.DataFrame, label_column: str, drop_columns: List[str],
                   drop_duplicates: bool = True) -> pd.DataFrame:
    """Run the full cleaning sequence and return a clean, numeric-feature DataFrame."""
    df = df.copy()
    df = drop_irrelevant_columns(df, drop_columns)
    df = coerce_numeric_features(df, label_column)
    df = replace_infinities(df)
    df = drop_missing(df, label_column)
    if drop_duplicates:
        df = drop_duplicate_rows(df)
    df = drop_constant_columns(df, label_column)
    df = df.reset_index(drop=True)
    logger.info("Clean dataset: %d rows, %d cols (%d feature cols + label)",
                len(df), df.shape[1], df.shape[1] - 1)
    return df
