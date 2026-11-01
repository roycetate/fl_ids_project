"""
knn_graph.py
------------
Build an undirected k-NN similarity graph over flow feature vectors.
Each flow is a node; edges connect nearest neighbors in scaled feature space.
"""

from typing import Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(
    x: np.ndarray,
    k: int,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an undirected k-NN graph.

    Parameters
    ----------
    x : (N, F) node feature matrix (already scaled).
    k : number of neighbors per node.
    metric : "euclidean" or "cosine".

    Returns
    -------
    edge_index : (2, E) int64, PyG convention (source row, target row).
    edge_weight : (E,) float64, similarity in (0, 1]; closer neighbors weigh more.
    """
    n_nodes = x.shape[0]
    if n_nodes <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float64)

    k_eff = min(k, n_nodes - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=-1)
    nn.fit(x)
    distances, indices = nn.kneighbors(x)

    # Skip self (column 0); convert distance -> weight in (0, 1].
    src_list, dst_list, weight_list = [], [], []
    for i in range(n_nodes):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            w = 1.0 / (1.0 + float(dist))
            src_list.extend([i, j])
            dst_list.extend([j, i])
            weight_list.extend([w, w])

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_weight = np.array(weight_list, dtype=np.float64)

    # Deduplicate undirected edges, keeping the max weight per pair.
    edge_index, edge_weight = _dedupe_edges(edge_index, edge_weight)
    return edge_index, edge_weight


def _dedupe_edges(
    edge_index: np.ndarray, edge_weight: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse duplicate (u,v) / (v,u) pairs, keeping max weight."""
    pairs = edge_index.T
    canonical = np.sort(pairs, axis=1)
    unique_pairs, inverse = np.unique(canonical, axis=0, return_inverse=True)

    max_weights = np.zeros(len(unique_pairs), dtype=np.float64)
    np.maximum.at(max_weights, inverse, edge_weight)

    edge_index = unique_pairs.T
    edge_weight = max_weights
    return edge_index, edge_weight
