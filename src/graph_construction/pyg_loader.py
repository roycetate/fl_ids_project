"""
pyg_loader.py
-------------
Load saved .npz graph batches into PyTorch-Geometric Data objects.
"""

import os
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data


def load_graph_npz(path: str) -> Data:
    """Load a single graph batch from disk."""
    d = np.load(path)
    return Data(
        x=torch.from_numpy(d["x"]).float(),
        y=torch.from_numpy(d["y"]).long(),
        edge_index=torch.from_numpy(d["edge_index"]).long(),
        edge_weight=torch.from_numpy(d["edge_weight"]).float(),
    )


def list_graph_files(graphs_dir: str, prefix: str) -> List[str]:
    """Return sorted paths for all graphs matching `prefix_batch_*.npz`."""
    files = [
        os.path.join(graphs_dir, f)
        for f in os.listdir(graphs_dir)
        if f.startswith(f"{prefix}_batch_") and f.endswith(".npz")
    ]
    return sorted(files, key=lambda p: int(os.path.basename(p).split("_batch_")[1].split(".")[0]))


def load_client_graphs(graphs_dir: str, client_id: str) -> List[Data]:
    """Load all graph batches for one client (e.g. 'client_0')."""
    return [load_graph_npz(p) for p in list_graph_files(graphs_dir, client_id)]


def load_test_graphs(graphs_dir: str) -> List[Data]:
    """Load all test graph batches."""
    return [load_graph_npz(p) for p in list_graph_files(graphs_dir, "test")]


def load_all_train_graphs(graphs_dir: str) -> List[Data]:
    """Load every client_* graph batch (for centralized training)."""
    graphs = []
    prefixes = sorted(
        {f.split("_batch_")[0] for f in os.listdir(graphs_dir)
         if f.startswith("client_") and f.endswith(".npz")}
    )
    for prefix in prefixes:
        graphs.extend(load_client_graphs(graphs_dir, prefix))
    return graphs


def discover_client_ids(graphs_dir: str) -> List[str]:
    """Return sorted client IDs present in graphs_dir."""
    return sorted(
        {f.split("_batch_")[0] for f in os.listdir(graphs_dir)
         if f.startswith("client_") and f.endswith(".npz")}
    )


def load_labels_from_clients(clients_dir: str) -> np.ndarray:
    """Concatenate all training labels (for computing class weights)."""
    labels = []
    for fname in sorted(os.listdir(clients_dir)):
        if fname.endswith(".npz"):
            labels.append(np.load(os.path.join(clients_dir, fname))["y"])
    return np.concatenate(labels) if labels else np.array([], dtype=np.int64)
