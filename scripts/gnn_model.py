"""GCN baseline model for pIC50 regression."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class IC50GCN(nn.Module):
    """
    GCN baseline:
      GCNConv(input → 64) → ReLU
      GCNConv(64 → 64)    → ReLU
      GCNConv(64 → 64)    → ReLU
      global_mean_pool
      Linear(64 → 32) → ReLU
      Linear(32 → 1)
    """

    def __init__(self, node_features: int = 23):
        super().__init__()
        self.conv1 = GCNConv(node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        return self.lin2(x)
