"""
make_dummy_data.py
-------------------
Generates a small synthetic CSV that mimics the *shape* and known quirks of
a real CICIDS2017 file (leading-space columns, occasional inf/NaN, a
multi-class 'Label' column, duplicate rows). This lets you smoke-test the
preprocessing pipeline immediately, before the real dataset is downloaded.

This is test/dev tooling only -- swap in the real CSVs under data/raw/
before doing anything that needs real results.

Usage:
    python tests/make_dummy_data.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

FEATURE_NAMES = [
    " Destination Port", " Flow Duration", " Total Fwd Packets",
    " Total Backward Packets", "Total Length of Fwd Packets",
    " Total Length of Bwd Packets", " Fwd Packet Length Max",
    " Fwd Packet Length Min", " Fwd Packet Length Mean", " Flow Bytes/s",
    " Flow Packets/s", " Flow IAT Mean", " Flow IAT Std", " Fwd IAT Total",
    " Bwd IAT Total", "Active Mean", " Idle Mean",
]

CLASSES = ["BENIGN", "DoS Hulk", "PortScan", "DDoS", "FTP-Patator", "Bot"]
CLASS_WEIGHTS = [0.70, 0.10, 0.08, 0.06, 0.04, 0.02]  # BENIGN-dominant, like real CICIDS2017


def make_dummy_csv(path: str, n_rows: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)

    data = {name: rng.exponential(scale=1000, size=n_rows) for name in FEATURE_NAMES}
    df = pd.DataFrame(data)

    # Inject some inf and NaN, matching real CICIDS2017's messiness.
    inf_idx = rng.choice(n_rows, size=max(1, n_rows // 200), replace=False)
    df.loc[inf_idx, " Flow Bytes/s"] = np.inf
    nan_idx = rng.choice(n_rows, size=max(1, n_rows // 150), replace=False)
    df.loc[nan_idx, " Flow IAT Std"] = np.nan

    df[" Label"] = rng.choice(CLASSES, size=n_rows, p=CLASS_WEIGHTS)

    df["Flow ID"] = [f"flow-{i}" for i in range(n_rows)]
    df["Source IP"] = "10.0.0." + (rng.integers(1, 254, n_rows)).astype(str)
    df["Timestamp"] = pd.Timestamp("2017-07-03") + pd.to_timedelta(
        rng.integers(0, 86400, n_rows), unit="s"
    )

    # Duplicate a few rows on purpose, like real capture data.
    dup_rows = df.sample(n=max(1, n_rows // 100), random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows -> {path}")


if __name__ == "__main__":
    make_dummy_csv(
        os.path.join(config.RAW_DATA_DIR, "Monday-WorkingHours.pcap_ISCX.csv"),
        n_rows=4000, seed=1,
    )
    make_dummy_csv(
        os.path.join(config.RAW_DATA_DIR, "Tuesday-WorkingHours.pcap_ISCX.csv"),
        n_rows=3000, seed=2,
    )
    print("\nDummy CICIDS2017-shaped data ready in", config.RAW_DATA_DIR)
    print("Run: python -m src.data_preprocessing.pipeline")
