import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ...project.logger import logger
from ..config import clip_config


def plot_rdm_submatrix(rdm, subset_size=100):
    subset_rdm = rdm[:subset_size, :subset_size]
    plt.figure(figsize=(8, 6))
    sns.heatmap(subset_rdm, cmap="viridis", annot=False, square=True)
    plt.title("Subset of Representational Dissimilarity Matrix (RDM)", fontsize=16)
    plt.xlabel("Image Index", fontsize=14)
    plt.ylabel("Image Index", fontsize=14)
    plt.show()


def plot_rdm_distribution(rdm, bins=30, exclude_diagonal=True):  # noqa: FBT002
    if exclude_diagonal:
        mask = ~np.eye(rdm.shape[0], dtype=bool)
        rdm_values = rdm[mask]
    else:
        rdm_values = rdm.flatten()

    plt.figure(figsize=(6, 4))
    plt.hist(rdm_values, bins=bins, color="blue", alpha=0.7)
    plt.title("Distribution of RDM Values", fontsize=16)
    plt.xlabel("RDM Value", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.show()


def show_image_pair(idx1, idx2, image_list, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image_list[idx1])
    axes[0].set_title("Image 1")

    axes[1].imshow(image_list[idx2])
    axes[1].set_title("Image 2")

    fig.suptitle(title, fontsize=14)
    plt.show()


def plot_training_history(train_loss, train_acc, test_loss, test_acc, metric="r2", std_dev=False, **kwargs):  # noqa: PLR0913, ANN003, FBT002
    epochs = range(1, len(train_loss) + 1)

    # Determine x label based on what kind of training, either standard or K-fold
    # Standard training plots over epochs, whereas K-fold plots over each fold (takes last epoch value)
    x_label = "Epoch" if not std_dev else "Fold"

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
    plt.xlabel(f"{x_label}", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title(f"Loss over {x_label}s", fontsize=16)
    plt.legend()
    plt.xticks(epochs)

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
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(metric.capitalize(), fontsize=14)
    plt.title(f"{metric.capitalize()} over {x_label}s", fontsize=16)
    plt.legend()
    plt.xticks(epochs)

    plt.tight_layout()
    plt.show()


def plot_accuracy_vs_layers(hidden_layers_list, accuracy_list, is_thingsvision=False, metric="r2"):  # noqa: FBT002
    if is_thingsvision:
        title = f"Metric = {metric}, Feature vectors: THINGSvision"
    else:
        title = f"Metric = {metric}, Feature vectors: CLIP Embeddings"
    plt.figure(figsize=(8, 6))
    plt.bar(hidden_layers_list, accuracy_list, color="blue")
    plt.xlabel("Number of Hidden Layers in the MLP Models", fontsize=14)
    plt.ylabel(f"{metric} Accuracy", fontsize=14)
    plt.ylim(top=1)
    plt.title(title, fontsize=16)
    plt.xticks(hidden_layers_list, fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def all_plots(train_loss, train_acc, test_loss, test_acc):
    if not clip_config.K_FOLD:
        logger.info("Standard historical plotting over training course (not KFold)")
        plot_training_history(
            train_loss, train_acc, test_loss, test_acc, metric=clip_config.ACCURACY
        )  # e.g. r2, spearman, pearson
    else:
        logger.info("KFold historical plotting over training course with stdev")
        train_loss_std = np.std(train_loss, axis=0)
        train_acc_std = np.std(train_acc, axis=0)
        test_loss_std = np.std(test_loss, axis=0)
        test_acc_std = np.std(test_acc, axis=0)

        plot_training_history(
            train_loss,
            train_acc,
            test_loss,
            test_acc,
            metric=clip_config.ACCURACY,
            std_dev=True,
            train_loss_std=train_loss_std,
            train_acc_std=train_acc_std,
            test_loss_std=test_loss_std,
            test_acc_std=test_acc_std,
        )


# used in deprectated model
def plot_rdms(true_rdm, predicted_rdm):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(true_rdm, cmap="viridis", origin="upper")
    axes[0].set_title("True RDM", fontsize=16)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(predicted_rdm, cmap="viridis", origin="upper")
    axes[1].set_title("Predicted RDM", fontsize=16)
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()
