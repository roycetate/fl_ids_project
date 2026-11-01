"""
Evaluation metrics: macro-F1, per-class F1, confusion matrix, classification report.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    graphs: List[Data],
    device: torch.device,
    batch_size: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on graph batches; return (y_true, y_pred) over all nodes."""
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    y_true, y_pred = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, getattr(batch, "edge_weight", None))
        preds = logits.argmax(dim=-1)
        y_true.append(batch.y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Return macro-F1, weighted-F1, per-class F1, and sklearn classification report."""
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))
    weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels))
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)

    per_class_dict = {}
    for i, lbl in enumerate(labels):
        name = class_names[int(lbl)] if class_names and int(lbl) < len(class_names) else str(lbl)
        per_class_dict[name] = float(per_class[i])

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=[
            class_names[int(l)] if class_names and int(l) < len(class_names) else str(l)
            for l in labels
        ] if class_names else None,
        zero_division=0,
        output_dict=True,
    )

    return {
        "macro_f1": macro,
        "weighted_f1": weighted,
        "per_class_f1": per_class_dict,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [int(l) for l in labels],
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    out_path: Optional[str] = None,
    normalize: bool = True,
) -> None:
    """Save a confusion-matrix heatmap (raw counts or row-normalized)."""
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm, row_sums, where=row_sums > 0, out=np.zeros_like(cm, dtype=float))
        fmt, title = ".2f", "Confusion Matrix (row-normalized)"
    else:
        cm_plot = cm
        fmt, title = "d", "Confusion Matrix"

    tick_labels = [
        class_names[int(l)] if class_names and int(l) < len(class_names) else str(l)
        for l in labels
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.5)))
    sns.heatmap(
        cm_plot, annot=len(labels) <= 20, fmt=fmt, cmap="Blues",
        xticklabels=tick_labels, yticklabels=tick_labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_metrics(metrics: Dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
