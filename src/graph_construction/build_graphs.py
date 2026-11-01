"""
build_graphs.py
---------------
Turn per-client feature shards into batched k-NN graphs saved as .npz files.

Usage:
    python -m src.graph_construction.build_graphs
    python -m src.graph_construction.build_graphs --k 10 --batch-size 2000
    python -m src.graph_construction.build_graphs --oversample-minority --stratified-batches
"""

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.graph_construction.knn_graph import build_knn_graph
from src.utils.seed import set_all_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_graphs")


def parse_args():
    p = argparse.ArgumentParser(description="Build k-NN graphs from FL client shards")
    p.add_argument("--clients-dir", default=config.CLIENTS_DIR)
    p.add_argument("--artifacts-dir", default=config.ARTIFACTS_DIR)
    p.add_argument("--out-dir", default=config.GRAPHS_DIR)
    p.add_argument("--k", type=int, default=config.KNN_K)
    p.add_argument("--metric", choices=["euclidean", "cosine"], default=config.KNN_METRIC)
    p.add_argument("--batch-size", type=int, default=config.GRAPH_BATCH_SIZE)
    p.add_argument(
        "--oversample-minority",
        action="store_true",
        default=config.OVERSAMPLE_MINORITY,
        help="Randomly oversample minority classes in each client shard before graphing.",
    )
    p.add_argument(
        "--stratified-batches",
        action="store_true",
        default=config.STRATIFIED_GRAPH_BATCHES,
        help="Spread classes across graph batches instead of using a fully random chunk order.",
    )
    p.add_argument("--seed", type=int, default=config.GRAPH_SEED)
    return p.parse_args()


def _oversample_minority_classes(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic feature interpolation for minority classes."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) <= 1:
        return X, y

    max_count = int(counts.max())
    target = min(15000, max_count)
    new_x, new_y = [X], [y]

    for cls, count in zip(classes, counts):
        if count < target and count >= 2:
            cls_indices = np.flatnonzero(y == cls)
            cls_x = X[cls_indices]
            n_needed = target - count

            idx1 = rng.choice(len(cls_x), size=n_needed, replace=True)
            idx2 = rng.choice(len(cls_x), size=n_needed, replace=True)
            lam = rng.uniform(0.05, 0.95, size=(n_needed, 1)).astype(np.float32)

            synth_x = cls_x[idx1] * lam + cls_x[idx2] * (1.0 - lam)
            synth_y = np.full(n_needed, cls, dtype=np.int64)

            new_x.append(synth_x)
            new_y.append(synth_y)

    res_x = np.vstack(new_x)
    res_y = np.concatenate(new_y)
    perm = rng.permutation(len(res_y))
    return res_x[perm], res_y[perm]


def _stratified_order(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return an order that interleaves classes, preserving deterministic randomness."""
    class_indices = []
    for cls in np.unique(y):
        indices = np.flatnonzero(y == cls)
        rng.shuffle(indices)
        class_indices.append(indices)

    ordered = []
    max_len = max(len(indices) for indices in class_indices)
    for offset in range(max_len):
        for indices in class_indices:
            if offset < len(indices):
                ordered.append(indices[offset])

    return np.array(ordered, dtype=np.int64)


def _batch_and_build(
    X: np.ndarray,
    y: np.ndarray,
    prefix: str,
    out_dir: str,
    k: int,
    metric: str,
    batch_size: int,
    rng: np.random.Generator,
    stratified_batches: bool = False,
) -> int:
    """Shuffle, chunk into batches, build a graph per chunk. Returns graph count."""
    n = len(y)
    perm = _stratified_order(y, rng) if stratified_batches else rng.permutation(n)
    X, y = X[perm], y[perm]

    count = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = X[start:end]
        y_batch = y[start:end]

        edge_index, edge_weight = build_knn_graph(x_batch, k=k, metric=metric)
        out_path = os.path.join(out_dir, f"{prefix}_batch_{count}.npz")
        np.savez_compressed(
            out_path,
            x=x_batch.astype(np.float32),
            y=y_batch.astype(np.int64),
            edge_index=edge_index.astype(np.int64),
            edge_weight=edge_weight.astype(np.float32),
        )
        logger.info(
            "  %s: %d nodes, %d edges -> %s",
            prefix, len(y_batch), edge_index.shape[1], os.path.basename(out_path),
        )
        count += 1
    return count


def run(args):
    set_all_seeds(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Client shards
    client_files = sorted(
        f for f in os.listdir(args.clients_dir) if f.endswith(".npz")
    )
    if not client_files:
        raise FileNotFoundError(
            f"No client .npz files in {args.clients_dir}. "
            "Run `python -m src.data_preprocessing.pipeline` first."
        )

    logger.info(
        "Building graphs for %d clients (k=%d, batch=%d, oversample=%s, stratified=%s)",
        len(client_files), args.k, args.batch_size,
        args.oversample_minority, args.stratified_batches,
    )
    for fname in client_files:
        client_id = fname.replace(".npz", "")
        data = np.load(os.path.join(args.clients_dir, fname))
        X, y = data["X"], data["y"]
        if args.oversample_minority:
            before = len(y)
            X, y = _oversample_minority_classes(X, y, rng)
            logger.info("  %s: oversampled %d -> %d training nodes", client_id, before, len(y))
        _batch_and_build(
            X, y, client_id, args.out_dir,
            args.k, args.metric, args.batch_size, rng, args.stratified_batches,
        )

    # Centralized test set
    test_path = os.path.join(args.artifacts_dir, "test_set.npz")
    if os.path.exists(test_path):
        logger.info("Building test graphs")
        test = np.load(test_path)
        _batch_and_build(
            test["X"], test["y"], "test", args.out_dir,
            args.k, args.metric, args.batch_size, rng, False,
        )
    else:
        logger.warning("No test_set.npz found at %s — skipping test graphs", test_path)

    logger.info("Graph construction complete -> %s", args.out_dir)


if __name__ == "__main__":
    run(parse_args())
