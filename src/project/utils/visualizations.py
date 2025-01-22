import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_rdm_submatrix(rdm, subset_size=100):
    subset_rdm = rdm[:subset_size, :subset_size]
    plt.figure(figsize=(8, 6))
    sns.heatmap(subset_rdm, cmap="viridis", annot=False, square=True)
    plt.title("Subset of Representational Dissimilarity Matrix (RDM)")
    plt.xlabel("Image Index")
    plt.ylabel("Image Index")
    plt.show()


def plot_rdm_distribution(rdm, bins=30, exclude_diagonal=True):  # noqa: FBT002
    if exclude_diagonal:
        mask = ~np.eye(rdm.shape[0], dtype=bool)
        rdm_values = rdm[mask]
    else:
        rdm_values = rdm.flatten()

    plt.figure(figsize=(6, 4))
    plt.hist(rdm_values, bins=bins, color="blue", alpha=0.7)
    plt.title("Distribution of RDM Values")
    plt.xlabel("RDM Value")
    plt.ylabel("Frequency")
    plt.show()


def plot_rdms(true_rdm, predicted_rdm):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(true_rdm, cmap="viridis", origin="upper")
    axes[0].set_title("True RDM")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(predicted_rdm, cmap="viridis", origin="upper")
    axes[1].set_title("Predicted RDM")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()


def plot_training_history(train_loss, train_acc, test_loss, test_acc, metric="r2", std_dev=False, **kwargs):  # noqa: PLR0913, ANN003, FBT002
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, test_loss, label="Testing Loss")
    if std_dev:
        plt.fill_between(
            epochs,
            train_loss - kwargs.get("train_loss_std", 0),
            train_loss + kwargs.get("train_loss_std", 0),
            alpha=0.2,
        )
        plt.fill_between(
            epochs, test_loss - kwargs.get("test_loss_std", 0), test_loss + kwargs.get("test_loss_std", 0), alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, test_acc, label="Testing Accuracy")
    if std_dev:
        plt.fill_between(
            epochs, train_acc - kwargs.get("train_acc_std", 0), train_acc + kwargs.get("train_acc_std", 0), alpha=0.2
        )
        plt.fill_between(
            epochs, test_acc - kwargs.get("test_acc_std", 0), test_acc + kwargs.get("test_acc_std", 0), alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_accuracy_vs_layers(hidden_layers_list, accuracy_list, metric="r2"):
    plt.figure(figsize=(8, 6))
    plt.bar(hidden_layers_list, accuracy_list, color="skyblue")
    plt.xlabel("Number of Layers in MLP")
    plt.ylabel(metric)
    plt.title(f"{metric} vs. Number of Layers")
    plt.xticks(hidden_layers_list)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_r2_from_nested_dict(data):
    """Plots a grouped bar chart of R^2 accuracy values from a nested dictionary structure.

    Args:
        data (dict): A nested dictionary where outer keys are groups,
                     inner keys are additional labels, and values are R^2 values.
                     Example: {
                         'MLP-0': {500: 0.8, 1000: 0.9},
                         'MLP-1': {500: 0.85, 1000: 0.88}
                     }
    """
    # Prepare data for plotting
    groups = list(data.keys())
    nested_keys = list(next(iter(data.values())).keys())  # Assume all groups have the same inner keys
    n_nested = len(nested_keys)
    group_positions = range(len(groups))
    bar_width = 0.8 / n_nested  # Width of each bar

    # Colors and positions for bars
    colors = plt.cm.tab10.colors[:n_nested]  # Use a colormap for distinct colors
    positions = [
        [group_pos + i * bar_width - (bar_width * (n_nested - 1) / 2) for group_pos in group_positions]
        for i in range(n_nested)
    ]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each nested key as a group of bars
    for i, nested_key in enumerate(nested_keys):
        r2_values = [data[group][nested_key] for group in groups]
        plt.bar(positions[i], r2_values, bar_width, label=f"{nested_key}", color=colors[i])

    # Add a red horizontal line at y=1
    plt.axhline(y=1, color="red", linestyle="--", linewidth=1.5, label="y = 1")

    # Labeling
    plt.xlabel("Models")
    plt.ylabel("R^2 Accuracy")
    plt.title("R^2 Accuracy for Different Configurations")
    plt.xticks(group_positions, groups)  # Set group names on x-axis
    plt.legend(title="Nested Keys")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Final layout adjustments
    plt.tight_layout()
    plt.show()
