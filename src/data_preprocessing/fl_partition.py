"""
fl_partition.py
----------------
Splits the (already scaled) training set across N simulated client
organizations, for use with the Flower federated learning setup in a
later phase.

Two strategies:
  - "iid":     each client gets a random, roughly equal, similarly-
               distributed shard. Good as a sanity-check baseline.
  - "non_iid": each client gets a skewed class distribution, drawn via a
               Dirichlet distribution per class (standard approach in FL
               literature, e.g. Hsu et al. 2019). This is the more
               realistic setting for cross-enterprise data, since
               different organizations see different mixes of attacks.

Test set is kept centralized (not partitioned) so the global model can be
evaluated consistently across FL rounds.
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def _iid_partition(y: np.ndarray, num_clients: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    return [shard for shard in np.array_split(indices, num_clients)]


def _non_iid_dirichlet_partition(
    y: np.ndarray, num_clients: int, alpha: float, seed: int
) -> List[np.ndarray]:
    """
    For each class, draw a Dirichlet(alpha) proportion vector over clients
    and assign that class's samples accordingly. Lower alpha => more skew
    (some clients barely see certain attack types, mimicking real
    cross-enterprise heterogeneity).
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        # Convert proportions to cumulative split points over this class's indices.
        split_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        cls_splits = np.split(cls_idx, split_points)

        for client_id, split in enumerate(cls_splits):
            client_indices[client_id].extend(split.tolist())

    return [np.array(sorted(idx)) for idx in client_indices]


def partition_clients(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    strategy: str = "non_iid",
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns a dict: {"client_0": {"X": ..., "y": ...}, "client_1": {...}, ...}
    """
    if strategy == "iid":
        shards = _iid_partition(y, num_clients, seed)
    elif strategy == "non_iid":
        shards = _non_iid_dirichlet_partition(y, num_clients, dirichlet_alpha, seed)
    else:
        raise ValueError(f"Unknown partition strategy: '{strategy}' (use 'iid' or 'non_iid')")

    clients = {}
    for i, idx in enumerate(shards):
        if len(idx) == 0:
            logger.warning("client_%d received 0 samples -- consider raising "
                            "dirichlet_alpha or lowering num_clients", i)
        clients[f"client_{i}"] = {"X": X[idx], "y": y[idx]}
        class_counts = np.bincount(y[idx]) if len(idx) else np.array([])
        logger.info("client_%d: %6d samples, class counts (by encoded label): %s",
                     i, len(idx), dict(enumerate(class_counts)))

    return clients
