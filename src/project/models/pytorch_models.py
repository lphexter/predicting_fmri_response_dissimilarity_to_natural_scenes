# models/pytorch_models.py

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

# not used in main.py - for initial testing, kept for tracking
class NeuralNetwork(nn.Module):
    """Simple MLP: 1024 -> 512 -> 1, with optional hidden layers, final activation if needed."""

    def __init__(self, hidden_layers=1, activation_func="linear"):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation_func = activation_func

        self.layers = nn.ModuleList()
        # Input (1024) -> 512
        self.layers.append(nn.Linear(1024, 512))

        # Optional hidden layers (512->512)
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(512, 512))

        # Output (512->1)
        self.layers.append(nn.Linear(512, 1))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        x = self.layers[-1](x)

        if self.activation_func == "sigmoid":
            return torch.sigmoid(x) * 2
        return x


class DynamicLayerSizeNeuralNetwork(nn.Module):
    """MLP where each hidden layer halves the size from the previous layer.

    e.g. 1024 -> 512 -> 256 -> ... -> 1
    """

    def __init__(self, hidden_layers=1, activation_func="linear"):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation_func = activation_func
        self.layers = nn.ModuleList()

        in_features = 1024
        out_features = 512
        self.layers.append(nn.Linear(in_features, out_features))

        current_size = out_features
        for _ in range(hidden_layers):
            next_size = current_size // 2
            self.layers.append(nn.Linear(current_size, next_size))
            current_size = next_size

        self.layers.append(nn.Linear(current_size, 1))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        x = self.layers[-1](x)

        if self.activation_func == "sigmoid":
            return torch.sigmoid(x) * 2
        return x
