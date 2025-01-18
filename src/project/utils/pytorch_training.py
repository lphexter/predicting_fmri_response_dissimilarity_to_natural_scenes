import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from .pytorch_data import PairDataset


def compute_accuracy(predictions, targets, metric="r2"):
    pred_np = predictions.detach().cpu().numpy()
    targ_np = targets.detach().cpu().numpy()

    if metric == "r2":
        return r2_score(targ_np, pred_np)
    if metric == "pearson":
        return pearsonr(targ_np, pred_np)[0]
    if metric == "spearman":
        return spearmanr(targ_np, pred_np)[0]
    raise ValueError(f"Unknown accuracy metric: {metric}")  # noqa: TRY003, EM102


def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10, metric="r2"):  # noqa: PLR0913
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    model.to(device)

    for epoch in range(num_epochs):
        ################
        #   TRAIN
        ################
        model.train()
        running_train_loss = 0.0
        running_train_acc = 0.0
        total_train_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)  # noqa: PLW2901
            y_batch = y_batch.to(device)  # noqa: PLW2901

            optimizer.zero_grad()
            preds = model(x_batch).squeeze(1)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            batch_size_ = x_batch.size(0)
            running_train_loss += loss.item() * batch_size_
            running_train_acc += compute_accuracy(preds, y_batch, metric) * batch_size_
            total_train_samples += batch_size_

        epoch_train_loss = running_train_loss / total_train_samples
        epoch_train_acc = running_train_acc / total_train_samples

        ################
        #   EVAL
        ################
        model.eval()
        running_test_loss = 0.0
        running_test_acc = 0.0
        total_test_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)  # noqa: PLW2901
                y_batch = y_batch.to(device)  # noqa: PLW2901

                preds = model(x_batch).squeeze(1)
                loss = criterion(preds, y_batch)

                batch_size_ = x_batch.size(0)
                running_test_loss += loss.item() * batch_size_
                running_test_acc += compute_accuracy(preds, y_batch, metric) * batch_size_
                total_test_samples += batch_size_

        epoch_test_loss = running_test_loss / total_test_samples
        epoch_test_acc = running_test_acc / total_test_samples

        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)
        test_loss_history.append(epoch_test_loss)
        test_acc_history.append(epoch_test_acc)

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"Train Loss: {epoch_train_loss:.4f}, {metric}: {epoch_train_acc:.4f} | "
            f"Test  Loss: {epoch_test_loss:.4f}, {metric}: {epoch_test_acc:.4f}"
        )

    return train_loss_history, train_acc_history, test_loss_history, test_acc_history


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
