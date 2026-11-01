"""
diagnose.py  (Phase 1)
----------------------
Train or load a centralized model and print per-class F1 + confusion matrix.
This is the diagnostic step before applying imbalance interventions.

Usage:
    python -m src.training.diagnose
    python -m src.training.diagnose --checkpoint results/centralized_model.pt
    python -m src.training.diagnose --epochs 30 --no-plot   # headless
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.graph_construction.pyg_loader import load_all_train_graphs, load_labels_from_clients, load_test_graphs
from src.training.metrics import collect_predictions, compute_metrics, plot_confusion_matrix, save_metrics
from src.training.trainer import (
    build_model,
    evaluate_model,
    get_device,
    infer_num_classes,
    infer_num_features,
    load_class_names,
    run_centralized_training,
    train_model,
)
from src.utils.seed import set_all_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnose")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: per-class F1 + confusion matrix diagnostics")
    p.add_argument("--graphs-dir", default=config.GRAPHS_DIR)
    p.add_argument("--clients-dir", default=config.CLIENTS_DIR)
    p.add_argument("--artifacts-dir", default=config.ARTIFACTS_DIR)
    p.add_argument("--out-dir", default=config.RESULTS_DIR)
    p.add_argument("--checkpoint", default=None, help="Load weights instead of training fresh")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--dropout", type=float, default=config.DROPOUT)
    p.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="ce")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_all_seeds(args.seed)
    device = get_device()

    train_graphs = load_all_train_graphs(args.graphs_dir)
    test_graphs = load_test_graphs(args.graphs_dir)
    class_names = load_class_names(args.artifacts_dir)
    y_train = load_labels_from_clients(args.clients_dir)

    in_ch = infer_num_features(train_graphs)
    num_classes = infer_num_classes(args.clients_dir, class_names)
    model = build_model(in_ch, num_classes, dropout=args.dropout)

    if args.checkpoint and os.path.exists(args.checkpoint):
        import torch
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        logger.info("Loaded checkpoint: %s", args.checkpoint)
    else:
        logger.info("Training baseline model for diagnosis (%d epochs, loss=%s)",
                    args.epochs, args.loss)
        model = train_model(
            model, train_graphs, y_train, num_classes, device,
            epochs=args.epochs, lr=args.lr, loss_type=args.loss, seed=args.seed,
        )

    metrics = evaluate_model(
        model, test_graphs, class_names, device,
        out_dir=args.out_dir, tag="diagnose",
    )

    y_true, y_pred = collect_predictions(model, test_graphs, device)
    if not args.no_plot:
        plot_confusion_matrix(
            y_true, y_pred, class_names,
            out_path=os.path.join(args.out_dir, "confusion_matrix.png"),
            normalize=True,
        )
        plot_confusion_matrix(
            y_true, y_pred, class_names,
            out_path=os.path.join(args.out_dir, "confusion_matrix_counts.png"),
            normalize=False,
        )
        logger.info("Saved confusion matrices to %s", args.out_dir)

    # Print a compact table for copy-paste into the paper
    print("\n--- Per-class F1 (diagnostic) ---")
    print(f"{'Class':<30} {'F1':>8}")
    print("-" * 40)
    for cls, f1 in sorted(metrics["per_class_f1"].items(), key=lambda x: -x[1]):
        print(f"{cls:<30} {f1:8.4f}")
    print(f"\nmacro-F1: {metrics['macro_f1']:.4f}")
    print(f"weighted-F1: {metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
