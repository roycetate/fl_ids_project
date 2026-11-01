"""
run_ablation.py
---------------
Run baseline + each imbalance intervention independently and log macro-F1
for the ablation table in the paper.

Usage:
    python -m src.training.run_ablation
    python -m src.training.run_ablation --modes centralized --losses ce weighted_ce focal
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.training.trainer import run_centralized_training
from src.utils.seed import set_all_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ablation")


def parse_args():
    p = argparse.ArgumentParser(description="Ablation: baseline vs imbalance interventions")
    p.add_argument("--graphs-dir", default=config.GRAPHS_DIR)
    p.add_argument("--clients-dir", default=config.CLIENTS_DIR)
    p.add_argument("--artifacts-dir", default=config.ARTIFACTS_DIR)
    p.add_argument("--out-dir", default=config.RESULTS_DIR)
    p.add_argument("--tag-prefix", default="centralized")
    p.add_argument("--losses", nargs="+", default=["ce", "weighted_ce", "focal"])
    p.add_argument("--dropouts", nargs="+", type=float, default=[config.DROPOUT])
    p.add_argument("--lrs", nargs="+", type=float, default=[config.LEARNING_RATE])
    p.add_argument("--weight-decays", nargs="+", type=float, default=[config.WEIGHT_DECAY])
    p.add_argument("--focal-gamma", type=float, default=config.FOCAL_GAMMA)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for loss in args.losses:
        for dropout in args.dropouts:
            for lr in args.lrs:
                for weight_decay in args.weight_decays:
                    tag = (
                        f"{args.tag_prefix}_loss-{loss}_dropout-{dropout}"
                        f"_lr-{lr}_wd-{weight_decay}"
                    )
                    logger.info("=== Ablation run: %s ===", tag)
                    _, metrics = run_centralized_training(
                        graphs_dir=args.graphs_dir,
                        clients_dir=args.clients_dir,
                        artifacts_dir=args.artifacts_dir,
                        out_dir=os.path.join(args.out_dir, tag),
                        loss_type=loss,
                        dropout=dropout,
                        lr=lr,
                        weight_decay=weight_decay,
                        epochs=args.epochs,
                        focal_gamma=args.focal_gamma,
                        seed=args.seed,
                    )
                    row = {
                        "tag": tag,
                        "graphs_dir": args.graphs_dir,
                        "loss": loss,
                        "dropout": dropout,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "macro_f1": metrics["macro_f1"],
                        "weighted_f1": metrics["weighted_f1"],
                        "per_class_f1": metrics["per_class_f1"],
                    }
                    results.append(row)
                    logger.info("  -> macro-F1 = %.4f", metrics["macro_f1"])

    summary_path = os.path.join(
        args.out_dir,
        f"ablation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Ablation summary saved to %s", summary_path)

    print("\n--- Ablation summary ---")
    print(f"{'Tag':<48} {'Loss':<15} {'Dropout':>8} {'LR':>8} {'WD':>8} {'macro-F1':>10}")
    print("-" * 106)
    for r in results:
        print(
            f"{r['tag']:<48} {r['loss']:<15} {r['dropout']:8.2f} "
            f"{r['lr']:8.4f} {r['weight_decay']:8.4f} {r['macro_f1']:10.4f}"
        )


if __name__ == "__main__":
    main()
