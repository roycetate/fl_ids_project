"""
feature_engineering.py
-----------------------
Turns the cleaned DataFrame into model-ready numpy arrays:
  - multi-class label encoding (BENIGN + every attack type kept separate,
    so the severity/response stages downstream can key off attack type)
  - stratified train/test split (stratified so rare attack classes still
    appear in both splits)
  - feature scaling (StandardScaler fit on train only, applied to test)

Returns numpy arrays plus the fitted encoder/scaler so they can be reused
identically at inference time.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def encode_labels(df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, LabelEncoder]:
    encoder = LabelEncoder()
    df = df.copy()
    df[label_column] = df[label_column].astype(str).str.strip()
    df["label_encoded"] = encoder.fit_transform(df[label_column])

    counts = df[label_column].value_counts()
    logger.info("Label distribution (%d classes):", len(encoder.classes_))
    for cls, n in counts.items():
        logger.info("  %-28s %8d  (%.3f%%)", cls, n, 100 * n / len(df))
    return df, encoder


def split_train_test(
    df: pd.DataFrame,
    label_column: str,
    test_size: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split. Falls back to a non-stratified split with a warning
    if any class has fewer members than needed for stratification."""
    class_counts = df["label_encoded"].value_counts()
    min_class_count = class_counts.min()
    can_stratify = min_class_count >= 2

    if not can_stratify:
        logger.warning(
            "At least one class has < 2 samples; falling back to a "
            "non-stratified split (class: %s, count: %d).",
            class_counts.idxmin(), min_class_count,
        )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df["label_encoded"] if can_stratify else None,
    )
    logger.info("Train/test split: %d train rows, %d test rows", len(train_df), len(test_df))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list]:
    """Fit StandardScaler on train features only; transform both splits.

    Returns X_train, y_train, X_test, y_test, fitted_scaler, feature_names.
    """
    feature_cols = [c for c in train_df.columns if c not in (label_column, "label_encoded")]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values.astype(np.float64))
    X_test = scaler.transform(test_df[feature_cols].values.astype(np.float64))

    y_train = train_df["label_encoded"].values.astype(np.int64)
    y_test = test_df["label_encoded"].values.astype(np.int64)

    logger.info("Scaled features: %d feature columns", len(feature_cols))
    return X_train, y_train, X_test, y_test, scaler, feature_cols
