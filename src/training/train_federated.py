"""
train_federated.py
------------------
Federated GAT training with Flower FedAvg across client graph batches.
Uses Flower NumPyClient + FedAvg aggregation (no Ray dependency).

Usage:
    python -m src.training.train_federated --strategy non_iid
    python -m src.training.train_federated --strategy iid --loss weighted_ce --rounds 10
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.graph_construction.pyg_loader import (
    discover_client_ids,
    load_client_graphs,
    load_test_graphs,
)
from src.models.gat import GATClassifier
from src.training.losses import build_loss
from src.training.metrics import collect_predictions, compute_metrics, save_metrics
from src.training.trainer import (
    build_model,
    get_device,
    infer_num_classes,
    infer_num_features,
    load_class_names,
    train_one_epoch,
)
from src.utils.seed import set_all_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_federated")


def _model_to_ndarrays(model: GATClassifier) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _ndarrays_to_model(model: GATClassifier, params: List[np.ndarray]) -> GATClassifier:
    state_dict = OrderedDict(
        {k: torch.tensor(v) for (k, _), v in zip(model.state_dict().items(), params)}
    )
    model.load_state_dict(state_dict, strict=True)
    return model


class GATFlowerClient(NumPyClient):
    def __init__(
        self,
        client_id: str,
        graphs: List[Data],
        model: GATClassifier,
        y_local: np.ndarray,
        num_classes: int,
        device: torch.device,
        local_epochs: int,
        lr: float,
        weight_decay: float,
        loss_type: str,
        focal_gamma: float,
        batch_size: int,
    ):
        self.client_id = client_id
        self.graphs = graphs
        self.model = model
        self.y_local = y_local
        self.num_classes = num_classes
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.batch_size = batch_size

    def get_parameters(self, config_dict):
        return _model_to_ndarrays(self.model)

    def fit(self, parameters, config_dict):
        self.model = _ndarrays_to_model(self.model, parameters).to(self.device)
        criterion = build_loss(
            self.loss_type, self.y_local, self.num_classes, self.device, self.focal_gamma
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loader = DataLoader(self.graphs, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.local_epochs):
            train_one_epoch(self.model, loader, optimizer, criterion, self.device)

        return _model_to_ndarrays(self.model), len(self.y_local), {}

    def evaluate(self, parameters, config_dict):
        self.model = _ndarrays_to_model(self.model, parameters).to(self.device)
        self.model.eval()
        loader = DataLoader(self.graphs, batch_size=self.batch_size, shuffle=False)
        criterion = build_loss(
            self.loss_type, self.y_local, self.num_classes, self.device, self.focal_gamma
        )
        total_loss, total_nodes = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch.x, batch.edge_index, getattr(batch, "edge_weight", None))
                total_loss += criterion(logits, batch.y).item() * batch.num_nodes
                total_nodes += batch.num_nodes
        return float(total_loss / max(total_nodes, 1)), len(self.y_local), {}


def _build_clients(
    client_ids: List[str],
    graphs_dir: str,
    in_ch: int,
    num_classes: int,
    device: torch.device,
    local_epochs: int,
    lr: float,
    weight_decay: float,
    loss_type: str,
    focal_gamma: float,
    batch_size: int,
    dropout: float,
) -> List[GATFlowerClient]:
    clients = []
    for cid in client_ids:
        graphs = load_client_graphs(graphs_dir, cid)
        y_local = np.concatenate([g.y.numpy() for g in graphs])
        model = build_model(in_ch, num_classes, dropout=dropout)
        clients.append(GATFlowerClient(
            cid, graphs, model, y_local, num_classes, device,
            local_epochs, lr, weight_decay, loss_type, focal_gamma, batch_size,
        ))
    return clients


def _fedavg_aggregate(fit_results: List[FitRes]):
    """Weighted average of client model weights (by number of training samples)."""
    weights = [r.num_examples for r in fit_results]
    total = sum(weights)
    param_lists = [parameters_to_ndarrays(r.parameters) for r in fit_results]
    aggregated = [
        sum(w * arr for w, arr in zip(weights, layer)) / total
        for layer in zip(*param_lists)
    ]
    return ndarrays_to_parameters(aggregated)


def run_federated_training(
    graphs_dir: str = config.GRAPHS_DIR,
    clients_dir: str = config.CLIENTS_DIR,
    artifacts_dir: str = config.ARTIFACTS_DIR,
    out_dir: str = config.RESULTS_DIR,
    num_rounds: int = config.FL_ROUNDS,
    local_epochs: int = config.LOCAL_EPOCHS,
    strategy_label: str = "non_iid",
    **train_kwargs,
) -> Tuple[GATClassifier, Dict]:
    """
    Flower-compatible FedAvg loop (NumPyClient + FedAvg.aggregate_fit).
    `strategy_label` is metadata; actual IID/non-IID comes from preprocessing.
    """
    os.makedirs(out_dir, exist_ok=True)
    set_all_seeds(train_kwargs.get("seed", config.RANDOM_SEED))
    device = get_device()

    client_ids = discover_client_ids(graphs_dir)
    sample_graphs = load_client_graphs(graphs_dir, client_ids[0])
    test_graphs = load_test_graphs(graphs_dir)
    class_names = load_class_names(artifacts_dir)

    in_ch = infer_num_features(sample_graphs)
    num_classes = infer_num_classes(clients_dir, class_names)
    dropout = train_kwargs.get("dropout", config.DROPOUT)
    loss_type = train_kwargs.get("loss_type", config.LOSS_TYPE)
    lr = train_kwargs.get("lr", config.LEARNING_RATE)
    weight_decay = train_kwargs.get("weight_decay", config.WEIGHT_DECAY)
    focal_gamma = train_kwargs.get("focal_gamma", config.FOCAL_GAMMA)

    clients = _build_clients(
        client_ids, graphs_dir, in_ch, num_classes, device,
        local_epochs, lr, weight_decay, loss_type, focal_gamma,
        config.TRAIN_BATCH_SIZE, dropout,
    )

    global_model = build_model(in_ch, num_classes, dropout=dropout)
    parameters = ndarrays_to_parameters(_model_to_ndarrays(global_model))

    logger.info(
        "Federated (%s): %d clients, %d rounds, %d local epochs/round, device=%s",
        strategy_label, len(clients), num_rounds, local_epochs, device,
    )

    param_arrays = parameters_to_ndarrays(parameters)
    for server_round in range(1, num_rounds + 1):
        fit_results = []
        for client in clients:
            updated, num_examples, metrics = client.fit(param_arrays, {})
            fit_results.append(FitRes(
                status=Status(code=Code.OK, message="success"),
                parameters=ndarrays_to_parameters(updated),
                num_examples=num_examples,
                metrics=metrics,
            ))

        aggregated = _fedavg_aggregate(fit_results)
        parameters = aggregated
        param_arrays = parameters_to_ndarrays(parameters)

        if server_round == 1 or server_round % max(1, num_rounds // 5) == 0 or server_round == num_rounds:
            eval_model = _ndarrays_to_model(
                build_model(in_ch, num_classes, dropout=dropout), param_arrays
            ).to(device)
            y_true, y_pred = collect_predictions(eval_model, test_graphs, device)
            from sklearn.metrics import f1_score
            macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
            logger.info("  round %3d/%d  test macro-F1=%.4f", server_round, num_rounds, macro)

    final_model = _ndarrays_to_model(
        build_model(in_ch, num_classes, dropout=dropout), param_arrays
    ).to(device)

    tag = f"federated_{strategy_label}"
    y_true, y_pred = collect_predictions(final_model, test_graphs, device)
    metrics = compute_metrics(y_true, y_pred, class_names)
    logger.info("=== %s (final) ===", tag)
    logger.info("  macro-F1: %.4f", metrics["macro_f1"])
    save_metrics(metrics, os.path.join(out_dir, f"{tag}_metrics.json"))
    torch.save(final_model.state_dict(), os.path.join(out_dir, f"{tag}_model.pt"))
    return final_model, metrics


def parse_args():
    p = argparse.ArgumentParser(description="Federated GAT training (Flower FedAvg)")
    p.add_argument("--graphs-dir", default=config.GRAPHS_DIR)
    p.add_argument("--clients-dir", default=config.CLIENTS_DIR)
    p.add_argument("--artifacts-dir", default=config.ARTIFACTS_DIR)
    p.add_argument("--out-dir", default=config.RESULTS_DIR)
    p.add_argument("--rounds", type=int, default=config.FL_ROUNDS)
    p.add_argument("--local-epochs", type=int, default=config.LOCAL_EPOCHS)
    p.add_argument("--strategy", choices=["iid", "non_iid"], default="non_iid",
                   help="Metadata label; actual partition comes from preprocessing")
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--dropout", type=float, default=config.DROPOUT)
    p.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default=config.LOSS_TYPE)
    p.add_argument("--focal-gamma", type=float, default=config.FOCAL_GAMMA)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _, metrics = run_federated_training(
        graphs_dir=args.graphs_dir,
        clients_dir=args.clients_dir,
        artifacts_dir=args.artifacts_dir,
        out_dir=args.out_dir,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        strategy_label=args.strategy,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
        seed=args.seed,
    )
    logger.info("Done. macro-F1 = %.4f", metrics["macro_f1"])
