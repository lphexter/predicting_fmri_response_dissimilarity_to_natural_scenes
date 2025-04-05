import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from ..config import clip_config
from ..logger import logger


def plot_rdm_submatrix(rdm, subset_size=100, binary=False):  # noqa: FBT002
    """Plots a subset of the Representational Dissimilarity Matrix (RDM).

    Args:
        rdm (numpy.ndarray): The full RDM matrix.
        subset_size (int, optional): The size of the subset to display.
        binary (boolean, optional): If a binary dissimilar/similar matrix, convert strings to integers for plotting.
    """
    subset_rdm = rdm[:subset_size, :subset_size]
    plt.figure(figsize=(8, 6))
    if binary:
        # Create a temporary numerical array for plotting
        subset_rdm = np.where(subset_rdm == "similar", 0, 1)
        cmap = mcolors.ListedColormap(["darkblue", "yellow"])
        sns.heatmap(
            subset_rdm,
            cmap=cmap,
            annot=False,
            square=True,
            cbar_kws={"ticks": [0.25, 0.75], "format": lambda x, _: {0.25: "similar", 0.75: "dissimilar"}.get(x, "")},
        )
    else:
        sns.heatmap(subset_rdm, cmap="viridis", annot=False, square=True)
    plt.title("Subset of Representational Dissimilarity Matrix (RDM)", fontsize=16)
    plt.xlabel("Image Index", fontsize=14)
    plt.ylabel("Image Index", fontsize=14)
    plt.show()


def plot_rdm_distribution(rdm, bins=30, exclude_diagonal=True):  # noqa: FBT002
    """Plots the distribution of values in the Representational Dissimilarity Matrix (RDM).

    Args:
        rdm (numpy.ndarray): The full RDM matrix.
        bins (int, optional): Number of bins in the histogram.
        exclude_diagonal (bool, optional): Whether to exclude diagonal values.
    """
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
    """Displays a pair of images side by side.

    Args:
        idx1 (int): Index of the first image in the list.
        idx2 (int): Index of the second image in the list.
        image_list (list of np.ndarray): List of images.
        title (str): Title for the image pair.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image_list[idx1])
    axes[0].set_title("Image 1")

    axes[1].imshow(image_list[idx2])
    axes[1].set_title("Image 2")

    fig.suptitle(title, fontsize=14)
    plt.show()


def plot_training_history(train_loss, train_acc, test_loss, test_acc, metric="r2", std_dev=False, **kwargs):  # noqa: PLR0913, ANN003, FBT002
    """Plots the training and testing loss and accuracy over epochs or folds.

    Args:
        train_loss (list): Training loss per epoch.
        train_acc (list): Training accuracy per epoch.
        test_loss (list): Testing loss per epoch.
        test_acc (list): Testing accuracy per epoch.
        metric (str, optional): Accuracy metric used (e.g., "r2", "pearson").
        std_dev (bool, optional): Whether to plot standard deviation.
        **kwargs: Additional arguments for standard deviation values.
    """
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
    """Plots model accuracy as a function of the number of hidden layers.

    Args:
        hidden_layers_list (list): List of hidden layer counts.
        accuracy_list (list): Corresponding model accuracies.
        is_thingsvision (bool, optional): Whether the model uses THINGSvision features.
        metric (str, optional): Accuracy metric used.
    """
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
    """Calls the appropriate training history plotting function based on training mode (standard or K-Fold).

    Args:
        train_loss (list): Training loss per epoch/fold.
        train_acc (list): Training accuracy per epoch/fold.
        test_loss (list): Testing loss per epoch/fold.
        test_acc (list): Testing accuracy per epoch/fold.
    """
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
    """Plots side-by-side heatmaps of the true and predicted Representational Dissimilarity Matrices (RDMs).

    Args:
        true_rdm (numpy.ndarray): The ground truth RDM.
        predicted_rdm (numpy.ndarray): The predicted RDM.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(true_rdm, cmap="viridis", origin="upper")
    axes[0].set_title("True RDM", fontsize=16)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(predicted_rdm, cmap="viridis", origin="upper")
    axes[1].set_title("Predicted RDM", fontsize=16)
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_and_metrics(y_true, y_pred, distribution_type="all"):
    """Plot the confusion matrix and evaluation metrics for an SVM classifier.

    Calculates accuracy, precision, recall, and F1 score using the provided true and predicted labels.
    Displays a heatmap of the confusion matrix with a textbox showing the computed metrics.

    Args:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        distribution_type (str, optional): Description of the data distribution. Defaults to "all".

    Returns:
        None
    """

    def calculate_metrics(y_true, y_pred):
        return (
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, pos_label="dissimilar"),
            recall_score(y_true, y_pred, pos_label="dissimilar"),
            f1_score(y_true, y_pred, pos_label="dissimilar"),
        )

    accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Similar", "Dissimilar"],
        yticklabels=["Similar", "Dissimilar"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    ax.collections[0].set_clim(vmin=0)
    metrics_text = f"Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1 Score: {f1:.3f}"
    plt.gcf().text(0.9, 0.5, metrics_text, fontsize=12, bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.8})
    plt.title(f"{distribution_type.title()} Pairs: Confusion Matrix and Metrics")
    plt.show()


def print_image_counts(labels):
    """Count and display the number of images for each label.

    Labels are interpreted as follows:
      - 0: Blue
      - 1: Red
      - 2: Green
      - -1: Unclassified

    Args:
        labels (array-like): Array of image labels.

    Returns:
        None
    """
    LABEL_TO_COLOR = {v: k for k, v in clip_config.COLOR_TO_LABEL.items()}
    unique, counts = np.unique(labels, return_counts=True)
    label_count_map = dict(zip(unique, counts, strict=False))
    logger.info("Image count by label:")
    for lbl, cnt in label_count_map.items():
        logger.info(f"{LABEL_TO_COLOR.get(lbl, 'Unknown')} ({lbl}): {cnt} images")


def display_images_by_label(images, labels, label_to_show=0, num_images=10):
    """Display a subset of images corresponding to a specific label.

    Args:
        images (list of np.ndarray): List of images.
        labels (array-like): Array of image labels.
        label_to_show (int, optional): Label of images to display (0, 1, 2, or -1). Defaults to 0.
        num_images (int, optional): Maximum number of images to display. Defaults to 10.

    Returns:
        None
    """
    indices = [i for i, lbl in enumerate(labels) if lbl == label_to_show]
    logger.info(f"\nDisplaying up to {num_images} images for label {label_to_show} (found {len(indices)})")
    sample_indices = indices[:num_images]
    plt.figure(figsize=(15, 5))
    for i, img_idx in enumerate(sample_indices):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[img_idx])
        plt.axis("off")
    plt.show()


def plot_fold_results(true_train, pred_train, true_test, pred_test, epoch):
    """Create scatter plots comparing true and predicted distances for training and test sets.

    Args:
        true_train (array-like): True distances for the training set.
        pred_train (array-like): Predicted distances for the training set.
        true_test (array-like): True distances for the test set.
        pred_test (array-like): Predicted distances for the test set.
        epoch (int): Epoch number (zero-indexed) used for titling the plots.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    overall_min_train = min(np.min(true_train), np.min(pred_train))
    overall_max_train = max(np.max(true_train), np.max(pred_train))
    overall_min_test = min(np.min(true_test), np.min(pred_test))
    overall_max_test = max(np.max(true_test), np.max(pred_test))

    axes[0].scatter(true_train, pred_train, alpha=0.5)
    axes[0].set_xlabel("True Distances")
    axes[0].set_ylabel("Predicted Distances")
    axes[0].set_title(f"[{clip_config.DISTRIBUTION_TYPE}] Train Epoch {epoch + 1}")
    axes[0].set_xlim(overall_min_train, overall_max_train)
    axes[0].set_ylim(overall_min_train, overall_max_train)

    axes[1].scatter(true_test, pred_test, alpha=0.5)
    axes[1].set_xlabel("True Distances")
    axes[1].set_ylabel("Predicted Distances")
    axes[1].set_title(f"[{clip_config.DISTRIBUTION_TYPE}] Test Epoch {epoch + 1}")
    axes[1].set_xlim(overall_min_test, overall_max_test)
    axes[1].set_ylim(overall_min_test, overall_max_test)

    plt.tight_layout()
    plt.show()


def plot_cv_summary(folds, fold_loss_list, fold_r2_list, chance_mse, shuffled_chance_mse):
    """Plot a summary of k-fold cross-validation performance for loss and R².

    Displays two plots:
      - Loss vs. Folds: Includes target loss, baseline (predicting the mean), and shuffled chance loss.
      - R² vs. Folds: Includes target R², baseline, and chance R² from shuffling.

    Args:
        folds (array-like): Array of fold indices.
        fold_loss_list (list): List of loss values for each fold.
        fold_r2_list (list): List of R² scores for each fold.
        chance_mse (float): Mean squared error when predicting the mean.
        shuffled_chance_mse (float): Mean squared error for shuffled predictions.

    Returns:
        None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{clip_config.DISTRIBUTION_TYPE.title()}: K-Fold Cross-Validation Summary", fontsize=16)

    ax1.axhline(y=0, color="green", linestyle="--", label="Target Loss")
    ax1.axhline(y=chance_mse, color="purple", linestyle="--", label="Predicting the mean")
    ax1.axhline(y=shuffled_chance_mse, color="red", linestyle="--", label="Chance (Shuffling values)")
    ax1.plot(folds, fold_loss_list, marker="o", color="tab:blue")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs. Folds")
    ax1.set_xticks(folds)
    ax1.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center")

    ax2.axhline(y=1, color="green", linestyle="--", label="Target R^2")
    ax2.axhline(y=0, color="purple", linestyle="--", label="Predicting the mean")
    ax2.axhline(y=-1, color="red", linestyle="--", label="Chance (Shuffling values)")
    ax2.plot(folds, fold_r2_list, marker="s", color="tab:blue")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("R^2")
    ax2.set_title("R^2 vs. Folds")
    ax2.set_xticks(folds)
    ax2.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center")

    plt.tight_layout()
    plt.show()
