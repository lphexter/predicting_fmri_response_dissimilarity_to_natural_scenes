import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from torch import optim
from torch.utils.data import DataLoader

from project.config.clip_config import ACCURACY, EPOCHS, HIDDEN_LAYERS, LEARNING_RATE
from project.models.pytorch_models import DynamicLayerSizeNeuralNetwork

from .pytorch_data import PairDataset


def compute_accuracy(predictions, targets, metric=ACCURACY):
    pred_np = predictions.detach().cpu().numpy()
    targ_np = targets.detach().cpu().numpy()

    if metric == "r2":
        return r2_score(targ_np, pred_np)
    if metric == "pearson":
        return pearsonr(targ_np, pred_np)[0]
    if metric == "spearman":
        return spearmanr(targ_np, pred_np)[0]
    raise ValueError(f"Unknown accuracy metric: {metric}")  # noqa: TRY003, EM102


def train_epoch(model, train_loader, criterion, optimizer, device):
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


def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=EPOCHS):  # noqa: PLR0913
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

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"Train Loss: {train_loss:.4f} {ACCURACY}: {train_accuracy:.4f} | "
            f"Test Loss: {test_loss:.4f} {ACCURACY}: {test_accuracy:.4f}"
        )

    return train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history


def train_model_kfold(loaders, criterion, device, num_layers=HIDDEN_LAYERS, num_epochs=EPOCHS):
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    for fold, (train_loader, test_loader) in enumerate(loaders):
        print(f"Fold {fold + 1}")
        model = DynamicLayerSizeNeuralNetwork(hidden_layers=num_layers)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        model.to(device)

        for epoch in range(num_epochs):
            train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_accuracy = validate_epoch(model, test_loader, criterion, device)

            print(
                f"Epoch {epoch + 1}/{num_epochs} "
                f"Train Loss: {train_loss:.4f} {ACCURACY}: {train_accuracy:.4f} | "
                f"Test Loss: {test_loss:.4f} {ACCURACY}: {test_accuracy:.4f}"
            )

        # appending the loss/accuracy metrics from the final epoch (assuming they're the best values)
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

    avg_test_accuracy = np.mean(test_accuracy_history)
    print(f"Average {ACCURACY} across folds: {avg_test_accuracy:.4f}")

    return train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history


def reconstruct_predicted_rdm(model, embeddings, row_indices, col_indices, device):
    N = embeddings.shape[0]
    predicted_rdm = np.zeros((N, N), dtype=np.float32)

    dummy_y = np.zeros(len(row_indices), dtype=np.float32)
    dataset = PairDataset(embeddings, (row_indices, col_indices), dummy_y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

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
