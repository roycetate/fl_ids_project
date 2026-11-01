"""
train_centralized.py
--------------------
Train the GAT on all client data combined (centralized upper bound).

Usage:
    python -m src.training.train_centralized
    python -m src.training.train_centralized --loss weighted_ce --dropout 0.4 --epochs 50
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.training.trainer import run_centralized_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_centralized")


def parse_args():
    p = argparse.ArgumentParser(description="Centralized GAT training on CICIDS2017 graphs")
    p.add_argument("--graphs-dir", default=config.GRAPHS_DIR)
    p.add_argument("--clients-dir", default=config.CLIENTS_DIR)
    p.add_argument("--artifacts-dir", default=config.ARTIFACTS_DIR)
    p.add_argument("--out-dir", default=config.RESULTS_DIR)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--dropout", type=float, default=config.DROPOUT)
    p.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default=config.LOSS_TYPE)
    p.add_argument("--focal-gamma", type=float, default=config.FOCAL_GAMMA)
    p.add_argument("--eval-every", type=int, default=0,
                   help="Evaluate every N epochs and optionally restore the best model.")
    p.add_argument("--select-best-metric", choices=["macro_f1", "weighted_f1"], default=None,
                   help="Metric to maximize when --eval-every is set.")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _, metrics = run_centralized_training(
        graphs_dir=args.graphs_dir,
        clients_dir=args.clients_dir,
        artifacts_dir=args.artifacts_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
        eval_every=args.eval_every,
        select_best_metric=args.select_best_metric,
        seed=args.seed,
    )
    logger.info("Done. macro-F1 = %.4f", metrics["macro_f1"])
