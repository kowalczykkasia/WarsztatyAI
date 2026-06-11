"""Improved GIN model with BatchNorm and Dropout for IC50 prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from torch_geometric.nn.norm import BatchNorm


class IC50GIN(nn.Module):
    """
    Graph Isomorphism Network with BatchNorm and Dropout.

    Architecture:
      GINConv(input → 128) → BatchNorm → ReLU → Dropout
      GINConv(128 → 128) → BatchNorm → ReLU → Dropout
      GINConv(128 → 128) → BatchNorm → ReLU → Dropout
      GINConv(128 → 128) → BatchNorm → ReLU → Dropout
      global_mean_pool
      Linear(128 → 64) → ReLU → Dropout
      Linear(64 → 1)
    """

    def __init__(self, node_features: int = 23, hidden_dim: int = 128, dropout_rate: float = 0.3):
        super().__init__()

        self.input_dim = node_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # GIN convolution layers with MLP
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(node_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )
        self.batch1 = BatchNorm(hidden_dim)

        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )
        self.batch2 = BatchNorm(hidden_dim)

        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )
        self.batch3 = BatchNorm(hidden_dim)

        self.conv4 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )
        self.batch4 = BatchNorm(hidden_dim)

        # MLP head
        self.lin1 = nn.Linear(hidden_dim, 64)
        self.lin2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GIN layers with BatchNorm and Dropout
        x = self.conv1(x, edge_index)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.batch3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = self.batch4(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # MLP head
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)

        return x
