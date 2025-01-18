# visualizations.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_rdm_submatrix(rdm, subset_size=100):
    """
    Plots a subset of the RDM (top-left corner).
    """
    subset_rdm = rdm[:subset_size, :subset_size]
    plt.figure(figsize=(8,6))
    sns.heatmap(subset_rdm, cmap='viridis', annot=False, square=True)
    plt.title("Subset of Representational Dissimilarity Matrix (RDM)")
    plt.xlabel("Image Index")
    plt.ylabel("Image Index")
    plt.show()

def plot_rdm_distribution(rdm, bins=30, exclude_diagonal=True):
    """
    Plots histogram of RDM values.
    """
    if exclude_diagonal:
        mask = ~np.eye(rdm.shape[0], dtype=bool)
        rdm_values = rdm[mask]
    else:
        rdm_values = rdm.flatten()

    plt.figure(figsize=(6,4))
    plt.hist(rdm_values, bins=bins, color='blue', alpha=0.7)
    plt.title("Distribution of RDM Values")
    plt.xlabel("RDM Value")
    plt.ylabel("Frequency")
    plt.show()

def plot_rdms(true_rdm, predicted_rdm):
    """
    Plots true RDM and predicted RDM side by side.
    """
    fig, axes = plt.subplots(1,2, figsize=(10,5))

    im0 = axes[0].imshow(true_rdm, cmap='viridis', origin='upper')
    axes[0].set_title("True RDM")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(predicted_rdm, cmap='viridis', origin='upper')
    axes[1].set_title("Predicted RDM")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()

def plot_training_history(train_loss, train_acc, test_loss, test_acc, metric="r2"):
    """
    Plots training vs. test loss and metric over epochs.
    """
    epochs_range = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label="Train Loss")
    plt.plot(epochs_range, test_loss,  label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()

    # Metric
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label=f"Train {metric}")
    plt.plot(epochs_range, test_acc,  label=f"Test {metric}")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} vs. Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_accuracy_vs_layers(hidden_layers_list, accuracy_list, metric='r2'):
    """
    Bar chart of accuracy vs. hidden layer count.
    """
    plt.figure(figsize=(8,6))
    plt.bar(hidden_layers_list, accuracy_list, color='skyblue')
    plt.xlabel("Number of Layers in MLP")
    plt.ylabel(metric)
    plt.title(f"{metric} vs. Number of Layers")
    plt.xticks(hidden_layers_list)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
