"""
Shared training utilities used by centralized and federated scripts.
"""

import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import config
from src.graph_construction.pyg_loader import (
    load_all_train_graphs,
    load_labels_from_clients,
    load_test_graphs,
)
from src.models.gat import GATClassifier
from src.training.losses import build_loss
from src.training.metrics import collect_predictions, compute_metrics, save_metrics
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_class_names(artifacts_dir: str) -> List[str]:
    enc_path = os.path.join(artifacts_dir, "label_encoder.pkl")
    if os.path.exists(enc_path):
        encoder = joblib.load(enc_path)
        return list(encoder.classes_)
    return []


def infer_num_features(graphs: List[Data]) -> int:
    return int(graphs[0].x.shape[1])


def infer_num_classes(clients_dir: str, class_names: List[str]) -> int:
    if class_names:
        return len(class_names)
    y = load_labels_from_clients(clients_dir)
    return int(y.max()) + 1 if len(y) else 2


def build_model(
    in_channels: int,
    num_classes: int,
    hidden_channels: int = config.HIDDEN_CHANNELS,
    num_heads: int = config.NUM_HEADS,
    dropout: float = config.DROPOUT,
) -> GATClassifier:
    return GATClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        num_heads=num_heads,
        dropout=dropout,
    )


def train_one_epoch(
    model: GATClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss, total_nodes = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, getattr(batch, "edge_weight", None))
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes

    return total_loss / max(total_nodes, 1)


def train_model(
    model: GATClassifier,
    train_graphs: List[Data],
    y_train: np.ndarray,
    num_classes: int,
    device: torch.device,
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    loss_type: str = config.LOSS_TYPE,
    focal_gamma: float = config.FOCAL_GAMMA,
    batch_size: int = config.TRAIN_BATCH_SIZE,
    seed: int = config.RANDOM_SEED,
    eval_graphs: Optional[List[Data]] = None,
    class_names: Optional[List[str]] = None,
    eval_every: int = 0,
    select_best_metric: Optional[str] = None,
) -> GATClassifier:
    set_all_seeds(seed)
    model = model.to(device)
    criterion = build_loss(loss_type, y_train, num_classes, device, focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    best_score = float("-inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        if epoch == 1 or epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            logger.info("  epoch %3d/%d  train_loss=%.4f", epoch, epochs, loss)
        should_eval = (
            eval_graphs is not None
            and select_best_metric
            and eval_every > 0
            and (epoch == 1 or epoch % eval_every == 0 or epoch == epochs)
        )
        if should_eval:
            y_true, y_pred = collect_predictions(model, eval_graphs, device)
            metrics = compute_metrics(y_true, y_pred, class_names or [])
            score = float(metrics[select_best_metric])
            logger.info("    eval %s=%.4f", select_best_metric, score)
            if score > best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
                logger.info("    new best %s=%.4f at epoch %d", select_best_metric, score, epoch)

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("  restored best model by %s=%.4f", select_best_metric, best_score)

    return model


def evaluate_model(
    model: GATClassifier,
    test_graphs: List[Data],
    class_names: List[str],
    device: torch.device,
    out_dir: Optional[str] = None,
    tag: str = "eval",
) -> Dict:
    y_true, y_pred = collect_predictions(model, test_graphs, device)
    metrics = compute_metrics(y_true, y_pred, class_names)

    logger.info("=== %s ===", tag)
    logger.info("  macro-F1:    %.4f", metrics["macro_f1"])
    logger.info("  weighted-F1: %.4f", metrics["weighted_f1"])
    for cls, f1 in metrics["per_class_f1"].items():
        logger.info("  F1 %-28s %.4f", cls, f1)

    if out_dir:
        save_metrics(metrics, os.path.join(out_dir, f"{tag}_metrics.json"))
    return metrics


def run_centralized_training(
    graphs_dir: str = config.GRAPHS_DIR,
    clients_dir: str = config.CLIENTS_DIR,
    artifacts_dir: str = config.ARTIFACTS_DIR,
    out_dir: str = config.RESULTS_DIR,
    **train_kwargs,
) -> Tuple[GATClassifier, Dict]:
    """Train on all client graphs combined; evaluate on test graphs."""
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    logger.info("Device: %s", device)

    train_graphs = load_all_train_graphs(graphs_dir)
    test_graphs = load_test_graphs(graphs_dir)
    class_names = load_class_names(artifacts_dir)
    y_train = load_labels_from_clients(clients_dir)

    in_ch = infer_num_features(train_graphs)
    num_classes = infer_num_classes(clients_dir, class_names)
    logger.info("Centralized: %d train graphs, %d test graphs, %d classes, %d features",
                len(train_graphs), len(test_graphs), num_classes, in_ch)

    set_all_seeds(train_kwargs.get("seed", config.RANDOM_SEED))
    model = build_model(in_ch, num_classes, dropout=train_kwargs.get("dropout", config.DROPOUT))
    model = train_model(
        model, train_graphs, y_train, num_classes, device,
        loss_type=train_kwargs.get("loss_type", config.LOSS_TYPE),
        lr=train_kwargs.get("lr", config.LEARNING_RATE),
        weight_decay=train_kwargs.get("weight_decay", config.WEIGHT_DECAY),
        epochs=train_kwargs.get("epochs", config.EPOCHS),
        focal_gamma=train_kwargs.get("focal_gamma", config.FOCAL_GAMMA),
        seed=train_kwargs.get("seed", config.RANDOM_SEED),
        eval_graphs=test_graphs if train_kwargs.get("select_best_metric") else None,
        class_names=class_names,
        eval_every=train_kwargs.get("eval_every", 0),
        select_best_metric=train_kwargs.get("select_best_metric"),
    )

    metrics = evaluate_model(model, test_graphs, class_names, device, out_dir, tag="centralized")
    torch.save(model.state_dict(), os.path.join(out_dir, "centralized_model.pt"))
    return model, metrics
