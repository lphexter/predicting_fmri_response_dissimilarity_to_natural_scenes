import random
import sys

import joblib
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.svm import SVC
from torch import nn, no_grad, optim
from torch.utils.data import DataLoader

from ..config import clip_config
from ..logger import logger
from ..models.pytorch_models import ContrastiveNetwork, DynamicLayerSizeNeuralNetwork, contrastive_loss
from .pytorch_data import (
    PairDataset,
    PairedData,
    PairedDataset,
    get_train_and_test_pairs,
    get_valid_pair_indices,
    make_pairs,
    prepare_data_for_kfold,
    train_test_split_pairs,
)
from .visualizations import plot_confusion_matrix_and_metrics, plot_cv_summary, plot_fold_results


def compute_accuracy(predictions, targets, metric=clip_config.ACCURACY):
    """Computes the accuracy of predictions based on the specified metric.

    Args:
        predictions (torch.Tensor): The model's predicted values.
        targets (torch.Tensor): The ground truth target values.
        metric (str, optional): The metric to use for accuracy calculation.

    Returns:
        float: The computed accuracy based on the specified metric.
    """
    pred_np = predictions.detach().cpu().numpy()
    targ_np = targets.detach().cpu().numpy()

    if metric == "r2":
        return r2_score(targ_np, pred_np)
    if metric == "pearson":
        return pearsonr(targ_np, pred_np)[0]
    if metric == "spearman":
        return spearmanr(targ_np, pred_np)[0]
    raise ValueError(f"Unknown accuracy metric: {metric}")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the model on (CPU/GPU).

    Returns:
        tuple with loss and accuracy average per epoch: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    count_samples = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)  # noqa: PLW2901
        y_batch = y_batch.to(device)  # noqa: PLW2901

        predictions = model(x_batch).squeeze(1)
        loss = criterion(predictions, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)
        count_samples += x_batch.size(0)

        accuracy_val = compute_accuracy(predictions, y_batch)
        running_accuracy += accuracy_val * x_batch.size(0)

    epoch_loss = running_loss / count_samples
    epoch_accuracy = running_accuracy / count_samples

    return epoch_loss, epoch_accuracy


def validate_epoch(model, test_loader, criterion, device):
    """Evaluates the model on the validation/test dataset.

    Args:
        model (torch.nn.Module): The trained neural network model.
        test_loader (torch.utils.data.DataLoader): DataLoader for validation/test data.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the model on (CPU/GPU).

    Returns:
        tuple with loss and accuracy average per epoch: (epoch_loss, epoch_accuracy)
    """
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    count_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)  # noqa: PLW2901
            y_batch = y_batch.to(device)  # noqa: PLW2901
            predictions = model(x_batch).squeeze(1)
            loss = criterion(predictions, y_batch)

            running_loss += loss.item() * x_batch.size(0)
            count_samples += x_batch.size(0)

            accuracy_val = compute_accuracy(predictions, y_batch)
            running_accuracy += accuracy_val * x_batch.size(0)

    epoch_loss = running_loss / count_samples
    epoch_accuracy = running_accuracy / count_samples

    return epoch_loss, epoch_accuracy


def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=clip_config.EPOCHS):  # noqa: PLR0913
    """Trains the model over multiple epochs and evaluates it at each epoch.

    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for validation/test data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the model on (CPU/GPU).
        num_epochs (int, optional): Number of training epochs.

    Returns:
        tuple: Lists containing loss and accuracy history.
    """
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    model.to(device)  # Move the model to the specified device

    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        # Validate for one epoch
        test_loss, test_accuracy = validate_epoch(model, test_loader, criterion, device)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"Train Loss: {train_loss:.4f} {clip_config.ACCURACY}: {train_accuracy:.4f} | "
            f"Test Loss: {test_loss:.4f} {clip_config.ACCURACY}: {test_accuracy:.4f}"
        )

    return train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history


def train_model_kfold(loaders, criterion, device, num_layers=clip_config.HIDDEN_LAYERS, num_epochs=clip_config.EPOCHS):
    """Trains the model using k-fold cross-validation.

    Args:
        loaders (list of tuples): A list containing (train_loader, test_loader) for each fold.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the model on (CPU/GPU).
        num_layers (int, optional): Number of hidden layers in the model.
        num_epochs (int, optional): Number of training epochs per fold.

    Returns:
        tuple: Lists containing loss and accuracy history across folds.
    """
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    for fold, (train_loader, test_loader) in enumerate(loaders):
        logger.info(f"Fold {fold + 1}")
        model = DynamicLayerSizeNeuralNetwork(hidden_layers=num_layers)
        optimizer = optim.Adam(model.parameters(), lr=clip_config.LEARNING_RATE)
        model.to(device)

        for epoch in range(num_epochs):
            train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_accuracy = validate_epoch(model, test_loader, criterion, device)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} "
                f"Train Loss: {train_loss:.4f} {clip_config.ACCURACY}: {train_accuracy:.4f} | "
                f"Test Loss: {test_loss:.4f} {clip_config.ACCURACY}: {test_accuracy:.4f}"
            )

        # appending the loss/accuracy metrics from the final epoch (assuming they're the best values)
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

    avg_test_accuracy = np.mean(test_accuracy_history)
    logger.info(f"Average {clip_config.ACCURACY} across folds: {avg_test_accuracy:.4f}")

    return train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history


def reconstruct_predicted_rdm(model, embeddings, row_indices, col_indices, device):
    """Reconstructs a predicted Representational Dissimilarity Matrix (RDM) using the trained model.

    Args:
        model (torch.nn.Module): The trained neural network model.
        embeddings (torch.Tensor): Embedding vectors.
        row_indices (numpy.ndarray): Row indices of the pairs.
        col_indices (numpy.ndarray): Column indices of the pairs.
        device (torch.device): Device to run the model on (CPU/GPU).

    Returns:
        numpy.ndarray: The reconstructed predicted RDM.
    """
    N = embeddings.shape[0]
    predicted_rdm = np.zeros((N, N), dtype=np.float32)

    dummy_y = np.zeros(len(row_indices), dtype=np.float32)
    dataset = PairDataset(embeddings, (row_indices, col_indices), dummy_y)
    loader = DataLoader(dataset, batch_size=clip_config.BATCH_SIZE, shuffle=False)

    model.eval()
    preds_list = []

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)  # noqa: PLW2901
            batch_preds = model(x_batch).squeeze(1)
            preds_list.append(batch_preds.cpu().numpy())

    all_preds = np.concatenate(preds_list)
    for i, (r, c) in enumerate(zip(row_indices, col_indices, strict=False)):
        predicted_rdm[r, c] = all_preds[i]
        predicted_rdm[c, r] = all_preds[i]

    return predicted_rdm


def train_all(row_indices, col_indices, rdm_values, embeddings, device, hidden_layers=clip_config.HIDDEN_LAYERS):  # noqa: PLR0913
    """Trains the model using either standard training or k-fold cross-validation.

    Args:
        row_indices (numpy.ndarray): Row indices of the pairs.
        col_indices (numpy.ndarray): Column indices of the pairs.
        rdm_values (numpy.ndarray): Target values for each pair.
        embeddings (torch.Tensor): Embedding vectors.
        device (torch.device): Device to run the model on (CPU/GPU).
        hidden_layers (int, optional): Number of hidden layers in the model.

    Returns:
        tuple: Lists containing loss and accuracy history.
    """
    criterion = nn.MSELoss()

    try:
        hidden_layers = int(hidden_layers)
    except ValueError as e:
        logger.error("Error converting hidden_layers to a number: %s", e)
        sys.exit(1)

    if not clip_config.K_FOLD:  # standard training
        #######################
        #    DATA PAIRS
        #######################
        logger.info("Starting standard training (not KFold)")

        X_train_indices, X_test_indices, y_train, y_test = train_test_split_pairs(
            row_indices,
            col_indices,
            rdm_values,
            test_size=clip_config.TEST_SIZE,  # e.g. 0.2 = 20% test
        )

        train_dataset = PairDataset(embeddings, X_train_indices, y_train)
        test_dataset = PairDataset(embeddings, X_test_indices, y_test)

        train_loader = DataLoader(
            train_dataset, batch_size=clip_config.BATCH_SIZE, shuffle=True
        )  # e.g. batch size of 32
        test_loader = DataLoader(test_dataset, batch_size=clip_config.BATCH_SIZE, shuffle=False)

        #######################
        #   MODEL + TRAIN
        #######################
        model = DynamicLayerSizeNeuralNetwork(
            hidden_layers=hidden_layers,
            activation_func=clip_config.ACTIVATION_FUNC,  # e.g. sigmoid, linear
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=clip_config.LEARNING_RATE)  # e.g. 0.1
        train_loss, train_acc, test_loss, test_acc = train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            num_epochs=clip_config.EPOCHS,  # e.g. 5 epochs
        )

    else:
        logger.info("Starting KFold Training")
        loaders = prepare_data_for_kfold(
            row_indices,
            col_indices,
            rdm_values,
            embeddings,
            n_splits=clip_config.K_FOLD_SPLITS,  # e.g. 5
        )
        train_loss, train_acc, test_loss, test_acc = train_model_kfold(
            loaders,
            criterion,
            device,
            num_layers=hidden_layers,  # e.g. 1 hidden layer
            num_epochs=clip_config.EPOCHS,
        )

    return train_loss, train_acc, test_loss, test_acc


#########################################
#    SVM METHODS
#########################################


def run_svm(embeddings, rdm, binary_rdm):
    """Train or load an SVM on paired data and evaluate its performance.

    This function performs the following steps:
      1. Retrieves valid pair indices based on the provided dissimilarity matrix (rdm) and a predefined distribution type.
      2. Randomizes the order of the valid pair indices.
      3. Splits the indices into training and testing sets.
      4. Constructs non-overlapping training and testing pairs.
      5. Creates paired data from embeddings for training and testing.
      6. Trains or loads an SVM classifier using the training data.
      7. Makes predictions on the testing data and plots the confusion matrix along with evaluation metrics.

    Args:
        embeddings (np.ndarray or torch.Tensor): The embeddings used to generate paired data.
        rdm (np.ndarray): A dissimilarity matrix.
        binary_rdm (np.ndarray): A binary dissimilarity matrix used for generating pairs.

    Returns:
        None
    """
    all_valid_indices = get_valid_pair_indices(rdm, distribution_type=clip_config.DISTRIBUTION_TYPE)
    shuffled_indices = all_valid_indices.copy()
    random.shuffle(shuffled_indices)

    train_indices, test_indices = train_test_split(np.arange(rdm.shape[0]), test_size=0.36, random_state=42)
    logger.info(f"Size of train indices: {train_indices.shape}, size of test indices: {test_indices.shape}")

    train_pairs, test_pairs = get_train_and_test_pairs(train_indices, test_indices, shuffled_indices)
    X_train, y_train_binary, _ = make_pairs(binary_rdm, rdm, train_pairs)
    X_test, y_test_binary, _ = make_pairs(binary_rdm, rdm, test_pairs)

    paired_data_train = PairedData(embeddings, X_train)
    paired_data_test = PairedData(embeddings, X_test)

    clf = train_or_load_svm(paired_data_train, y_train_binary)
    logger.info(f"Preparing to predict test data using clf: {clf}")
    y_test_pred = clf.predict(paired_data_test)
    plot_confusion_matrix_and_metrics(y_test_binary, y_test_pred, distribution_type=clip_config.DISTRIBUTION_TYPE)


def train_or_load_svm(
    X_train,  # noqa: N803
    y_train,
    model_to_load=clip_config.LOAD_SVM_MODEL_PATH,
    save_model_file_name=clip_config.SAVE_SVM_MODEL_PATH,
):
    """Train a new SVM classifier or load an existing one, and save the model if training occurs.

    If a model path is provided in `model_to_load`, the function attempts to load the classifier from that path.
    Otherwise, it trains a new SVM classifier using the provided training data and labels.
    The classifier is saved to `save_model_file_name` if a valid path is provided.

    Args:
        X_train (array-like): Training features for the SVM.
        y_train (array-like): Training labels corresponding to X_train.
        model_to_load (str, optional): File path to load a pre-trained model. Defaults to clip_config.LOAD_SVM_MODEL_PATH.
        save_model_file_name (str, optional): File path to save the trained model. Defaults to clip_config.SAVE_SVM_MODEL_PATH.

    Returns:
        clf: Trained or loaded SVM classifier.

    Raises:
        RuntimeError: If loading or saving the model fails.
        ValueError: If y_train does not contain at least two unique classes when training a new model.
    """
    if model_to_load != "":
        logger.info(f"Preparing to load model from path: {model_to_load}")
        try:
            clf = joblib.load(model_to_load)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_to_load}: {e}") from e
    else:
        logger.info("Training SVM from scratch")
        unique_classes = set(y_train)
        if len(unique_classes) < 2:  # noqa: PLR2004
            raise ValueError("Training failed: y_train must have at least two unique classes.")
        clf = SVC(degree=clip_config.DEGREE, kernel=clip_config.KERNEL)
        clf.fit(X_train, y_train)
        if save_model_file_name != "":
            logger.info(f"Preparing to save model to path: {save_model_file_name}")
            try:
                joblib.dump(clf, save_model_file_name)
            except Exception as e:
                raise RuntimeError(f"Failed to save model to {save_model_file_name}: {e}") from e

    return clf


#########################################
#    CONTRASTIVE LEARNING METHODS
#########################################


def compute_mse_chance_metrics(y_true):
    """Compute the baseline mean squared error (MSE) and a shuffled chance MSE.

    The baseline MSE is computed using the mean of y_true as the prediction.
    The shuffled chance MSE is computed by shuffling y_true and comparing it to the original values.

    Args:
        y_true (np.ndarray): Array of true values.

    Returns:
        tuple: A tuple containing:
            - chance_mse (float): The MSE when predicting the mean of y_true.
            - shuffled_chance_mse (float): The MSE when predicting shuffled values of y_true.
    """
    baseline_prediction = np.mean(y_true)
    chance_mse = np.mean((y_true - baseline_prediction) ** 2)

    y_shuffled = y_true.copy()
    np.random.shuffle(y_shuffled)
    shuffled_chance_mse = np.mean((y_true - y_shuffled) ** 2)

    return chance_mse, shuffled_chance_mse


def train_fold(model, optimizer, train_dataloader, test_dataloader, device):
    """Train the model for a single fold over a number of epochs and evaluate on the test set.

    For each epoch, the function trains the model using the training dataloader,
    computes training loss and R² score, and then evaluates the model on the test dataloader.
    It returns the final test metrics and the predictions for both training and testing sets.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        test_dataloader (DataLoader): DataLoader for the testing dataset.
        device (torch.device): Device on which the model is trained.

    Returns:
        tuple: A tuple containing:
            - last_epoch_test_metrics (tuple): Test loss and R² score from the last epoch.
            - all_true_train (list): True training values.
            - all_pred_train (list): Predicted training values.
            - all_true_test (list): True testing values.
            - all_pred_test (list): Predicted testing values.
    """
    last_epoch_test_metrics = None
    all_true_train, all_pred_train = None, None
    all_true_test, all_pred_test = None, None

    for epoch in range(clip_config.EPOCHS):
        model.train()
        epoch_loss_train = 0.0
        true_train = []
        pred_train = []
        for emb1, emb2, true_distance in train_dataloader:
            optimizer.zero_grad()
            predicted_distance, loss = contrastive_loss(
                model, emb1.to(device), emb2.to(device), true_distance.to(device)
            )
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item()
            true_train.extend(true_distance.cpu().numpy())
            pred_train.extend(predicted_distance.cpu().detach().numpy())

        r2_train = r2_score(true_train, pred_train)
        logger.info(
            f"Epoch {epoch + 1}, Train Loss: {epoch_loss_train / len(train_dataloader):.4f}, Train R^2: {r2_train:.4f}"
        )

        model.eval()
        epoch_loss_test = 0.0
        true_test = []
        pred_test = []
        with no_grad():
            for emb1, emb2, true_distance in test_dataloader:
                _, test_loss = contrastive_loss(model, emb1.to(device), emb2.to(device), true_distance.to(device))
                epoch_loss_test += test_loss.item()
                pred_dist = model(emb1, emb2).squeeze(1)
                true_test.extend(true_distance.cpu().numpy())
                pred_test.extend(pred_dist.cpu().detach().numpy())
        r2_test = r2_score(true_test, pred_test)
        logger.info(
            f"Epoch {epoch + 1}, Test Loss: {epoch_loss_test / len(test_dataloader):.4f}, Test R^2: {r2_test:.4f}"
        )

        last_epoch_test_metrics = (epoch_loss_test / len(test_dataloader), r2_test)
        all_true_train, all_pred_train = true_train, pred_train
        all_true_test, all_pred_test = true_test, pred_test

    return last_epoch_test_metrics, all_true_train, all_pred_train, all_true_test, all_pred_test


def run_kfold_cv(embeddings, binary_rdm, rdm, device, k=clip_config.K_FOLD_SPLITS):
    """K-fold cross-validation with Siamese Network

    Run k-fold cross-validation using a shuffled split, train a ContrastiveNetwork on each fold,
    and summarize the cross-validation performance.

    The function performs the following:
      - Retrieves valid pair indices and shuffles them.
      - Splits the data into k folds using a random split.
      - For each fold, builds training and testing pairs, creates paired datasets, and initializes
        a new ContrastiveNetwork model and optimizer.
      - Trains the model on the training set and evaluates on the test set.
      - Plots results for each fold and summarizes overall performance across folds.

    Args:
        embeddings (np.ndarray or torch.Tensor): The embeddings used for pairing.
        binary_rdm (np.ndarray): Binary dissimilarity matrix used for pairing.
        rdm (np.ndarray): Dissimilarity matrix.
        device (torch.device): Device on which to perform training.
        k (int, optional): Number of cross-validation folds. Defaults to clip_config.K_FOLD_SPLITS.

    Returns:
        None
    """
    all_valid_indices = get_valid_pair_indices(rdm, distribution_type=clip_config.DISTRIBUTION_TYPE)
    shuffled_indices = all_valid_indices.copy()
    random.shuffle(shuffled_indices)
    N_images = embeddings.shape[0]

    fold_r2_list = []
    fold_loss_list = []
    folds = np.arange(1, k + 1)
    ss = ShuffleSplit(n_splits=k, test_size=clip_config.TEST_SIZE, random_state=42)

    for fold_idx, (train_img_idx, test_img_idx) in enumerate(ss.split(np.arange(N_images))):
        logger.info(f"\n=== Fold {fold_idx + 1}/{k} ===")
        logger.info(f"Train image indices: {len(train_img_idx)}; Test image indices: {len(test_img_idx)}")
        logger.info(
            f"Train percentage: {round(len(train_img_idx) / N_images, 3)}; Test percentage: {round(len(test_img_idx) / N_images, 3)}"
        )

        train_pairs, test_pairs = get_train_and_test_pairs(train_img_idx, test_img_idx, shuffled_indices)
        total = len(train_pairs) + len(test_pairs)
        logger.info(f"Train pairs: {len(train_pairs)}; Test pairs: {len(test_pairs)}")
        logger.info(
            f"Train percentage: {round(len(train_pairs) / total, 3)}; Test percentage: {round(len(test_pairs) / total, 3)}"
        )

        X_train, _, y_train_numeric = make_pairs(binary_rdm, rdm, train_pairs)
        X_test, _, y_test_numeric = make_pairs(binary_rdm, rdm, test_pairs)

        train_dataset = PairedDataset(embeddings, X_train, y_train_numeric)
        train_dataloader = DataLoader(train_dataset, batch_size=clip_config.BATCH_SIZE, shuffle=True)
        test_dataset = PairedDataset(embeddings, X_test, y_test_numeric)
        test_dataloader = DataLoader(test_dataset, batch_size=clip_config.BATCH_SIZE, shuffle=False)

        model = ContrastiveNetwork(input_dim=embeddings.shape[1], dropout_percentage=clip_config.DROPOUT_PERCENTAGE).to(
            device
        )
        optimizer = optim.Adam(model.parameters(), lr=clip_config.LEARNING_RATE)

        test_metrics, true_train, pred_train, true_test, pred_test = train_fold(
            model, optimizer, train_dataloader, test_dataloader, device
        )
        fold_loss_list.append(test_metrics[0])
        fold_r2_list.append(test_metrics[1])
        plot_fold_results(true_train, pred_train, true_test, pred_test, clip_config.EPOCHS - 1)

    logger.info("\n=== Cross-Validation Summary ===")
    for i in range(k):
        logger.info(f"Fold {i + 1}: Test R^2 = {fold_r2_list[i]:.4f}, Test Loss = {fold_loss_list[i]:.4f}")

    mean_r2 = np.mean(fold_r2_list)
    std_r2 = np.std(fold_r2_list)
    mean_loss = np.mean(fold_loss_list)
    std_loss = np.std(fold_loss_list)

    logger.info(f"\nMean Test R^2: {mean_r2:.4f} ± {std_r2:.4f}")
    logger.info(f"Mean Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")

    chance_mse, shuffled_chance_mse = compute_mse_chance_metrics(y_test_numeric)
    plot_cv_summary(folds, fold_loss_list, fold_r2_list, chance_mse, shuffled_chance_mse)
