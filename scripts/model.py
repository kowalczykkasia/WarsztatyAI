"""MLP model for IC50 prediction."""

import torch
import torch.nn as nn


class IC50MLP(nn.Module):
    """
    Multi-Layer Perceptron for IC50 (pIC50) prediction.

    Architecture:
    2048 (Morgan fingerprint) -> 512 -> 128 -> 1 (pIC50 value)
    with ReLU activations and Dropout
    """

    def __init__(self, input_dim=2048, hidden_dims=[512, 128], dropout_rate=0.3):
        """
        Args:
            input_dim: Input feature dimension (Morgan fingerprint size)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate (default 0.3)
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers with ReLU and Dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer (regression: single value)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights with He initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with He initialization for ReLU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 2048)

        Returns:
            Output tensor of shape (batch_size, 1) - predicted pIC50 values
        """
        return self.network(x)
