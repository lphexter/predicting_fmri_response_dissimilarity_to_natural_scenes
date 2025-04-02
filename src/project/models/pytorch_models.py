# models/pytorch_models.py

import torch
import torch.nn.functional as F  # noqa: N812
from torch import cat, nn


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


# Define Contrastive Learning Network
class ContrastiveNetwork(nn.Module):
    def __init__(self, input_dim, dropout_percentage=0.5):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(p=dropout_percentage), nn.Linear(512, 256), nn.ReLU()
        )
        # Add a layer to compute similarity from the extracted features
        self.similarity_head = nn.Sequential(
            nn.Linear(256 * 2, 128),  # Takes concatenated features
            nn.ReLU(),
            nn.Linear(128, 1),  # Outputs a single similarity score
        )

    def forward(self, emb1, emb2):
        z1 = self.feature_extractor(emb1)
        z2 = self.feature_extractor(emb2)
        # Concatenate features and pass through similarity head
        merged = cat([z1, z2], dim=1)
        return self.similarity_head(merged)


# Contrastive Loss Function
def contrastive_loss(model, emb1, emb2, true_similarity):
    """Compute a custom loss function for similarity prediction"""
    # Get predicted similarity score from the model (modified)
    similarity_score = model(emb1, emb2).squeeze(1)  # The model already outputs the similarity score

    # Loss based on the difference between true similarity and predicted similarity
    loss = F.mse_loss(similarity_score, true_similarity)  # Use MSE to train similarity

    return similarity_score, loss  # Return the similarity score and loss
