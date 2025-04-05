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


class ContrastiveNetwork(nn.Module):
    """A neural network for contrastive learning

    Extracts features from input embeddings and computes a similarity score between pairs of embeddings.

    Attributes:
        feature_extractor (nn.Sequential): A sequential module that extracts features from input.
        similarity_head (nn.Sequential): A sequential module that computes the similarity score
                                           from concatenated feature pairs.
    """

    def __init__(self, input_dim, dropout_percentage=0.5):
        """Initialize the ContrastiveNetwork.

        Args:
            input_dim (int): Dimension of the input features.
            dropout_percentage (float, optional): Dropout probability. Defaults to 0.5.
        """
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(p=dropout_percentage), nn.Linear(512, 256), nn.ReLU()
        )
        self.similarity_head = nn.Sequential(nn.Linear(256 * 2, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, emb1, emb2):
        """Compute the similarity score between two input embeddings.

        Args:
            emb1 (torch.Tensor): The first input embedding.
            emb2 (torch.Tensor): The second input embedding.

        Returns:
            torch.Tensor: The computed similarity score.
        """
        z1 = self.feature_extractor(emb1)
        z2 = self.feature_extractor(emb2)
        merged = cat([z1, z2], dim=1)
        return self.similarity_head(merged)


def contrastive_loss(model, emb1, emb2, true_similarity):
    """Compute the mean squared error loss between the predicted and true similarity scores.

    Args:
        model (ContrastiveNetwork): The contrastive network model used to predict similarity.
        emb1 (torch.Tensor): The first input embedding.
        emb2 (torch.Tensor): The second input embedding.
        true_similarity (torch.Tensor): The ground truth similarity score.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The predicted similarity score.
            - torch.Tensor: The computed MSE loss.
    """
    similarity_score = model(emb1, emb2).squeeze(1)
    loss = F.mse_loss(similarity_score, true_similarity)
    return similarity_score, loss
