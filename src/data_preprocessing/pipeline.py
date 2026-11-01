"""
pipeline.py
-----------
End-to-end data preprocessing pipeline for the closed-loop FL-IDS project.

Stages:
    1. Load raw CICIDS2017 CSVs                (load_data.py)
    2. Clean (infinities, NaNs, dupes, junk)    (clean_data.py)
    3. Encode multi-class labels                (feature_engineering.py)
    4. Stratified train/test split              (feature_engineering.py)
    5. Scale features (fit on train only)       (feature_engineering.py)
    6. Partition train set across FL clients    (fl_partition.py)
    7. Save everything to data/processed/       (this file)

Usage:
    python -m src.data_preprocessing.pipeline
    python -m src.data_preprocessing.pipeline --num-clients 8 --strategy iid
    python -m src.data_preprocessing.pipeline --raw-dir /path/to/csvs

Run from the project root (the directory containing config.py).
"""

import argparse
import logging
import os
import sys

import joblib
import numpy as np

# Allow running as `python -m src.data_preprocessing.pipeline` from project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.data_preprocessing.load_data import load_raw_dataset
from src.data_preprocessing.clean_data import clean_dataset
from src.data_preprocessing.feature_engineering import (
    encode_labels,
    split_train_test,
    scale_features,
)
from src.data_preprocessing.fl_partition import partition_clients

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="CICIDS2017 preprocessing + FL partitioning pipeline")
    p.add_argument("--raw-dir", default=config.RAW_DATA_DIR,
                    help="Directory containing raw CICIDS2017 CSVs")
    p.add_argument("--out-dir", default=config.PROCESSED_DATA_DIR,
                    help="Directory to write processed outputs")
    p.add_argument("--num-clients", type=int, default=config.NUM_CLIENTS)
    p.add_argument("--strategy", choices=["iid", "non_iid"], default=config.PARTITION_STRATEGY)
    p.add_argument("--alpha", type=float, default=config.DIRICHLET_ALPHA,
                    help="Dirichlet alpha for non_iid partitioning (lower = more skewed)")
    p.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def run_pipeline(args):
    artifacts_dir = os.path.join(args.out_dir, "artifacts")
    clients_dir = os.path.join(args.out_dir, "clients")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(clients_dir, exist_ok=True)

    # 1. Load -----------------------------------------------------------
    logger.info("=== Stage 1/6: Loading raw data ===")
    raw_df = load_raw_dataset(args.raw_dir, config.CICIDS2017_FILES)

    # 2. Clean ------------------------------------------------------------
    logger.info("=== Stage 2/6: Cleaning ===")
    clean_df = clean_dataset(
        raw_df,
        label_column=config.LABEL_COLUMN,
        drop_columns=config.DROP_COLUMNS,
        drop_duplicates=config.DROP_DUPLICATES,
    )

    # 3. Encode labels ----------------------------------------------------
    logger.info("=== Stage 3/6: Encoding labels ===")
    encoded_df, label_encoder = encode_labels(clean_df, config.LABEL_COLUMN)

    # 4. Split --------------------------------------------------------------
    logger.info("=== Stage 4/6: Train/test split ===")
    train_df, test_df = split_train_test(
        encoded_df, config.LABEL_COLUMN, args.test_size, args.seed
    )

    # 5. Scale ------------------------------------------------------------
    logger.info("=== Stage 5/6: Scaling features ===")
    X_train, y_train, X_test, y_test, scaler, feature_names = scale_features(
        train_df, test_df, config.LABEL_COLUMN
    )

    # 6. Partition across FL clients --------------------------------------
    logger.info("=== Stage 6/6: Partitioning across %d FL clients (%s) ===",
                args.num_clients, args.strategy)
    clients = partition_clients(
        X_train, y_train,
        num_clients=args.num_clients,
        strategy=args.strategy,
        dirichlet_alpha=args.alpha,
        seed=args.seed,
    )

    # --- Save --------------------------------------------------------------
    logger.info("Saving artifacts to %s", artifacts_dir)
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(artifacts_dir, "label_encoder.pkl"))
    with open(os.path.join(artifacts_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))

    np.savez_compressed(os.path.join(artifacts_dir, "test_set.npz"), X=X_test, y=y_test)
    logger.info("  test_set.npz        -> X%s, y%s", X_test.shape, y_test.shape)

    logger.info("Saving %d client shards to %s", len(clients), clients_dir)
    for client_id, data in clients.items():
        out_path = os.path.join(clients_dir, f"{client_id}.npz")
        np.savez_compressed(out_path, X=data["X"], y=data["y"])
        logger.info("  %s.npz -> X%s, y%s", client_id, data["X"].shape, data["y"].shape)

    logger.info("Pipeline complete.")
    logger.info("Classes (%d): %s", len(label_encoder.classes_), list(label_encoder.classes_))
    return {
        "train_df_shape": train_df.shape,
        "test_df_shape": test_df.shape,
        "num_features": len(feature_names),
        "num_classes": len(label_encoder.classes_),
        "classes": list(label_encoder.classes_),
    }


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
