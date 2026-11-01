"""
load_data.py
------------
Locates raw CICIDS2017 CSV files and loads them into a single pandas
DataFrame with normalized column names.

The official CICIDS2017 release has two well-known quirks this module
handles automatically:
  1. Column names carry inconsistent leading/trailing whitespace
     (e.g. " Label", " Destination Port").
  2. Some files are encoded as latin-1 / cp1252 rather than utf-8.
"""

import os
import glob
import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names so 'Label' and ' Label' match."""
    df.columns = [c.strip() for c in df.columns]
    return df


def _read_single_csv(path: str) -> pd.DataFrame:
    """Read one CICIDS2017 CSV, trying utf-8 first and falling back to latin-1."""
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        logger.warning("utf-8 decode failed for %s, retrying with latin-1", path)
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df = _normalize_columns(df)
    df["__source_file__"] = os.path.basename(path)
    return df


def discover_raw_files(raw_dir: str, expected_files: Optional[List[str]] = None) -> List[str]:
    """
    Return the list of CSV paths actually present in raw_dir.

    If expected_files is given, only those filenames are considered (in that
    order); otherwise every *.csv in raw_dir is used. This lets the pipeline
    run on a partial download (e.g. just one day) without erroring out.
    """
    if expected_files:
        found = [os.path.join(raw_dir, f) for f in expected_files
                 if os.path.exists(os.path.join(raw_dir, f))]
        missing = [f for f in expected_files
                   if not os.path.exists(os.path.join(raw_dir, f))]
        if missing:
            logger.warning("%d expected file(s) not found in %s (skipping): %s",
                            len(missing), raw_dir, missing)
    else:
        found = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))

    if not found:
        raise FileNotFoundError(
            f"No CICIDS2017 CSV files found in '{raw_dir}'. "
            f"Download the dataset (e.g. from the UNB site or Kaggle 'cicids2017' "
            f"mirror) and place the CSVs there, or point RAW_DATA_DIR at the "
            f"correct folder in config.py."
        )
    return found


def load_raw_dataset(raw_dir: str, expected_files: Optional[List[str]] = None) -> pd.DataFrame:
    """Load and concatenate all discovered CICIDS2017 CSVs into one DataFrame."""
    paths = discover_raw_files(raw_dir, expected_files)
    logger.info("Loading %d raw file(s) from %s", len(paths), raw_dir)

    frames = []
    for path in paths:
        df = _read_single_csv(path)
        logger.info("  %-55s -> %8d rows, %3d cols", os.path.basename(path), len(df), df.shape[1])
        frames.append(df)

    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    logger.info("Combined dataset: %d rows, %d cols", len(combined), combined.shape[1])
    return combined
