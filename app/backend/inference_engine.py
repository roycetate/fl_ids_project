"""
Inference Engine for GAT Threat Detector.
Handles PyTorch model loading, k-NN graph construction from 70-D flow vectors,
Softmax multi-class threat prediction, and attention weight extraction.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.gat import GATClassifier

# Standard 13 CICIDS2017 multi-class attack categories
ATTACK_CLASSES = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack – Brute Force",
    "Web Attack – XSS",
    "Infiltration / Sql Inject"
]


class GATInferenceEngine:
    def __init__(self, in_channels: int = 70, num_classes: int = 13, k: int = 10):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.k = k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize GATClassifier with 128 channels & 4 heads
        self.model = GATClassifier(
            in_channels=in_channels,
            hidden_channels=128,
            num_classes=num_classes,
            num_heads=4,
            dropout=0.1
        ).to(self.device)
        self.model.eval()

    def build_knn_edges(self, x_tensor: torch.Tensor):
        """
        Constructs k-NN similarity graph from feature matrix X (N x F).
        Returns edge_index [2, E] and edge_weight [E].
        """
        N = x_tensor.shape[0]
        if N <= 1:
            # Single node graph with self loop
            edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
            edge_weight = torch.tensor([1.0], dtype=torch.float, device=self.device)
            return edge_index, edge_weight

        # Compute pairwise Euclidean distance matrix
        dist_matrix = torch.cdist(x_tensor, x_tensor, p=2.0)  # [N, N]
        
        k_neighbors = min(self.k, N - 1)
        # Find top-k nearest neighbors (excluding self distance 0)
        _, topk_indices = torch.topk(dist_matrix, k=k_neighbors + 1, largest=False, dim=1)
        
        src_nodes = []
        dst_nodes = []
        edge_weights = []

        for i in range(N):
            for neighbor_idx in topk_indices[i]:
                j = neighbor_idx.item()
                if i != j:
                    d_ij = dist_matrix[i, j].item()
                    s_ij = 1.0 / (1.0 + d_ij)
                    src_nodes.append(i)
                    dst_nodes.append(j)
                    edge_weights.append(s_ij)

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long, device=self.device)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float, device=self.device)
        return edge_index, edge_weight

    def predict_flow_batch(self, features_array: np.ndarray):
        """
        Predicts threat classes for numpy array of flow features (N x 70).
        Returns list of prediction dicts.
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(features_array, dtype=torch.float32, device=self.device)
            
            # Pad or trim features if not exactly 70
            N, F_in = x_tensor.shape
            if F_in < self.in_channels:
                padding = torch.zeros((N, self.in_channels - F_in), device=self.device)
                x_tensor = torch.cat([x_tensor, padding], dim=1)
            elif F_in > self.in_channels:
                x_tensor = x_tensor[:, :self.in_channels]

            edge_index, edge_weight = self.build_knn_edges(x_tensor)
            
            # GAT Forward pass
            logits = self.model(x_tensor, edge_index)
            probabilities = F.softmax(logits, dim=-1).cpu().numpy()
            predicted_indices = np.argmax(probabilities, axis=-1)

            results = []
            for idx in range(N):
                pred_class_idx = int(predicted_indices[idx])
                pred_class_name = ATTACK_CLASSES[pred_class_idx] if pred_class_idx < len(ATTACK_CLASSES) else f"Class_{pred_class_idx}"
                conf = float(probabilities[idx, pred_class_idx])
                
                # Get top 3 class probabilities
                top_3_idx = np.argsort(probabilities[idx])[::-1][:3]
                top_3 = [
                    {
                        "class": ATTACK_CLASSES[c_i] if c_i < len(ATTACK_CLASSES) else f"Class_{c_i}",
                        "probability": round(float(probabilities[idx, c_i]) * 100, 2)
                    }
                    for c_i in top_3_idx
                ]

                results.append({
                    "node_id": idx,
                    "predicted_class": pred_class_name,
                    "confidence": round(conf * 100, 2),
                    "is_malicious": pred_class_name != "BENIGN",
                    "top_3_predictions": top_3
                })

            return results, edge_index.cpu().numpy().tolist(), edge_weight.cpu().numpy().tolist()
