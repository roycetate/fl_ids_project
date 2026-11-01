"""
GAT classifier for multi-class intrusion detection on flow graphs.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GATClassifier(nn.Module):
    """
    Enhanced GAT classifier with multi-head attention, GELU activations,
    linear residual skip projection, and multi-layer node classification head.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_classes: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.conv2 = GATConv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=2,
            concat=True,
            dropout=dropout,
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.skip = nn.Linear(in_channels, hidden_channels * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x, edge_index, edge_weight=None):
        h_skip = self.skip(x)
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv1(h, edge_index)
        h = self.bn1(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.gelu(h + h_skip)
        return self.classifier(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
