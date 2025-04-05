import numpy as np
import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from project.config import clip_config
from project.utils import pytorch_training
from project.utils.pytorch_data import PairedData
from project.utils.pytorch_training import (
    compute_accuracy,
    compute_mse_chance_metrics,
    reconstruct_predicted_rdm,
    run_kfold_cv,
    run_svm,
    train_all,
    train_epoch,
    train_fold,
    train_model,
    train_or_load_svm,
    validate_epoch,
)


###############
## CONTRASTIVE SIAMESE NETWORK TESTS
###############
class DummyContrastive(nn.Module):
    """A dummy contrastive model that always returns a constant similarity score."""

    def __init__(self, constant=0.5):
        super().__init__()
        self.constant = constant
        self.dummy_param = nn.Parameter(torch.tensor(constant))

    def forward(self, emb1, _):
        batch_size = emb1.size(0)
        return self.dummy_param * torch.ones((batch_size, 1), device=emb1.device)


def get_dummy_contrastive_loader(num_samples=4, input_dim=10, batch_size=2):
    """Creates a dummy DataLoader for contrastive training with random tensors."""
    emb1 = torch.randn(num_samples, input_dim)
    emb2 = torch.randn(num_samples, input_dim)
    true_distance = torch.full((num_samples,), 0.5)
    dataset = torch.utils.data.TensorDataset(emb1, emb2, true_distance)
    return DataLoader(dataset, batch_size=batch_size)


def test_run_kfold_cv(monkeypatch):
    """Test run_kfold_cv executes without error using dummy data.

    Monkeypatch plotting functions to suppress output.
    """
    device = "cpu"
    # Create dummy embeddings for 5 images.
    embeddings = np.random.rand(5, 10)
    # Create dummy RDMs.
    rdm = np.random.rand(5, 5)
    binary_rdm = np.where(rdm > 0.5, "dissimilar", "similar")
    monkeypatch.setattr("project.utils.pytorch_training.plot_fold_results", lambda *_, **__: None)
    monkeypatch.setattr("project.utils.pytorch_training.plot_cv_summary", lambda *_, **__: None)
    # Run k-fold cv; we expect no errors.
    run_kfold_cv(embeddings, binary_rdm, rdm, device, k=2)


def test_compute_mse_chance_metrics():
    """Test that compute_mse_chance_metrics computes the expected baseline MSE."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    baseline = np.mean(y_true)
    expected_mse = np.mean((y_true - baseline) ** 2)
    chance_mse, shuffled_mse = compute_mse_chance_metrics(y_true)
    np.testing.assert_almost_equal(chance_mse, expected_mse, decimal=5)
    assert isinstance(shuffled_mse, float)


def test_train_fold():
    """Test train_fold runs without error and returns float metrics using a dummy contrastive model."""
    device = "cpu"
    model = DummyContrastive(constant=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = get_dummy_contrastive_loader(num_samples=4, input_dim=10, batch_size=2)
    test_loader = get_dummy_contrastive_loader(num_samples=4, input_dim=10, batch_size=2)
    metrics, _, _, _, _ = train_fold(model, optimizer, train_loader, test_loader, device)
    # Check that metrics are floats
    assert isinstance(metrics[0], float)
    assert isinstance(metrics[1], float)


#########################################
# SVM TESTS
#########################################


def test_train_or_load_svm_training(tmp_path):
    """Test train_or_load_svm when training a new SVM with dummy paired data."""
    # Create dummy embeddings and trivial pair indices.
    embeddings = np.random.rand(10, 5)
    pair_indices = [(i, i) for i in range(10)]
    paired_data = PairedData(embeddings, pair_indices)
    # Create dummy y_train with two unique classes.
    y_train = np.array([0, 1] * 5)
    # Force training by setting model_to_load empty.
    svm = train_or_load_svm(
        paired_data, y_train, model_to_load="", save_model_file_name=str(tmp_path / "svm_model.pkl")
    )
    predictions = svm.predict(paired_data)
    assert len(predictions) == len(paired_data)


@pytest.mark.usefixtures("tmp_path")
def test_run_svm(monkeypatch):
    """Test run_svm runs without error by monkeypatching SVM training and plotting functions."""
    embeddings = np.random.rand(10, 5)
    rdm = np.random.rand(10, 10)
    binary_rdm = np.where(rdm > 0.5, "dissimilar", "similar")

    # Override train_or_load_svm to return a dummy SVM that predicts zeros.
    class DummySVM:
        def predict(self, data):
            return np.zeros(len(data))

    monkeypatch.setattr("project.utils.pytorch_training.train_or_load_svm", lambda _data, _y: DummySVM())
    monkeypatch.setattr(
        "project.utils.pytorch_training.plot_confusion_matrix_and_metrics",
        lambda *_, **__: None,
    )
    run_svm(embeddings, rdm, binary_rdm)


###############
## OLD TESTS - MLP
###############


class DummyModel(nn.Module):
    """A dummy model that always returns a constant prediction.

    Its forward pass returns a tensor of shape (batch_size, 1)
    with a fixed value.
    """

    def __init__(self, constant_value=0.5):
        super().__init__()
        self.constant_value = constant_value
        # Adding a dummy parameter ensures that model.parameters() is non-empty.
        self.dummy_param = nn.Parameter(torch.tensor(constant_value))

    def forward(self, x):
        batch_size = x.shape[0]
        # Multiply the dummy parameter by a tensor of ones so that the output
        # is a function of the dummy parameter and gradients can flow.
        return self.dummy_param * torch.ones((batch_size, 1), device=x.device)


def get_dummy_loader(num_samples=8, input_dim=20, batch_size=2):
    """Create a dummy DataLoader using random inputs and targets."""
    # Random inputs of shape (num_samples, input_dim)
    inputs = torch.randn(num_samples, input_dim)
    # Random targets of shape (num_samples,)
    targets = torch.randn(num_samples)
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size)


def test_compute_accuracy_r2():
    """Test compute_accuracy with the R2 metric. (If predictions equal targets, r2 should be 1.0.)"""
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.0, 2.0, 3.0])
    acc = compute_accuracy(predictions, targets, metric="r2")
    assert abs(acc - 1.0) < 1e-5


def test_compute_accuracy_pearson():
    """Test compute_accuracy with the Pearson metric. (For identical arrays, the correlation should be 1.0.)"""
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.0, 2.0, 3.0])
    acc = compute_accuracy(predictions, targets, metric="pearson")
    assert abs(acc - 1.0) < 1e-5


def test_compute_accuracy_spearman():
    """Test compute_accuracy with the Spearman metric. (For identical arrays, the correlation should be 1.0.)"""
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.0, 2.0, 3.0])
    acc = compute_accuracy(predictions, targets, metric="spearman")
    assert abs(acc - 1.0) < 1e-5


def test_compute_accuracy_invalid():
    """Test that compute_accuracy raises a ValueError for an unknown metric."""
    predictions = torch.tensor([1.0, 2.0])
    targets = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError):  # noqa: PT011
        compute_accuracy(predictions, targets, metric="unknown_metric")


def test_train_and_validate_epoch():
    """Test that train_epoch and validate_epoch run without error and return floats.

    Uses a dummy model, dummy DataLoader, MSE loss, and Adam optimizer.
    """
    device = "cpu"
    model = DummyModel(constant_value=0.5)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loader = get_dummy_loader(num_samples=8, input_dim=20, batch_size=2)

    train_loss, train_acc = train_epoch(model, loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, loader, criterion, device)

    assert isinstance(train_loss, float)
    assert isinstance(train_acc, float)
    assert isinstance(val_loss, float)
    assert isinstance(val_acc, float)


def test_train_model(monkeypatch):
    """Test train_model with a dummy dataset and dummy model.

    Monkeypatch DynamicLayerSizeNeuralNetwork to return a DummyModel.
    """
    device = "cpu"
    # Create dummy train and test loaders.
    train_loader = get_dummy_loader(num_samples=8, input_dim=20, batch_size=2)
    test_loader = get_dummy_loader(num_samples=8, input_dim=20, batch_size=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(DummyModel().parameters(), lr=0.001)
    # Override the model creation with our DummyModel.
    monkeypatch.setattr(
        pytorch_training,
        "DynamicLayerSizeNeuralNetwork",
        lambda _hidden_layers, _activation_func=None: DummyModel(constant_value=0.5),
    )
    # Override EPOCHS for quick testing.
    monkeypatch.setattr(clip_config, "EPOCHS", 1)

    # Call train_model.
    train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = train_model(
        model=DummyModel(),
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=1,
    )
    # Verify that history lists have one element.
    assert len(train_loss_hist) == 1
    assert len(train_acc_hist) == 1
    assert len(test_loss_hist) == 1
    assert len(test_acc_hist) == 1


def test_train_model_kfold(monkeypatch):
    """Test train_model_kfold using dummy pair data and dummy loaded features.

    Monkeypatch the dynamic model to return our DummyModel.

    We simulate a scenario with 4 images, which gives 6 unique pairs:
        (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    This should help ensure that both the train and test splits (from KFold)
    contain at least one pair.
    """
    device = "cpu"
    # Dummy pair indices for 4 images (6 pairs):
    row_indices = np.array([0, 0, 0, 1, 1, 2])
    col_indices = np.array([1, 2, 3, 2, 3, 3])
    rdm_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # Create dummy loaded features for 4 images; each with 10 features.
    loaded_features = torch.tensor(np.random.rand(4, 10).astype(np.float32))
    # Use 2 folds for simplicity.
    monkeypatch.setattr(clip_config, "EPOCHS", 1)
    # Override the model creation to use DummyModel.
    monkeypatch.setattr(
        pytorch_training,
        "DynamicLayerSizeNeuralNetwork",
        lambda hidden_layers, activation_func=None: DummyModel(constant_value=0.5),  # noqa: ARG005
    )
    # Prepare k-fold data
    loaders = pytorch_training.prepare_data_for_kfold(row_indices, col_indices, rdm_values, loaded_features, n_splits=2)
    # Now train using k-fold.
    train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = pytorch_training.train_model_kfold(
        loaders, criterion=nn.MSELoss(), device=device, num_layers=1, num_epochs=1
    )
    # There should be 2 entries (one per fold).
    assert all(
        len(metric_history) == 2 for metric_history in (train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist)
    )


def test_reconstruct_predicted_rdm():
    """Test reconstruct_predicted_rdm with a dummy model that returns a constant prediction.

    Verify that the predicted RDM is symmetric and has the expected constant value off-diagonals.
    """
    device = "cpu"
    # Create dummy embeddings for 3 images; each with 10 features.
    embeddings = torch.tensor(np.random.rand(3, 10).astype(np.float32))
    # Define dummy pair indices: for a 3-image RDM, valid pair indices from generate_pair_indices.
    row_indices = np.array([0, 0, 1])
    col_indices = np.array([1, 2, 2])
    # Use our DummyModel that returns constant 0.5.
    model = DummyModel(constant_value=0.5).to(device)
    predicted_rdm = reconstruct_predicted_rdm(model, embeddings, row_indices, col_indices, device)
    # Check that predicted_rdm is symmetric.
    np.testing.assert_allclose(predicted_rdm, predicted_rdm.T)
    # For the pairs provided, the value should be 0.5.
    for r, c in zip(row_indices, col_indices, strict=False):
        assert abs(predicted_rdm[r, c] - 0.5) < 1e-5


def test_train_all_standard(monkeypatch):
    """Test train_all when K_FOLD is False.

    Generate dummy pair data for 4 images so that the train/test split
    (with test_size=0.5) produces at least one pair in each set.
    Monkeypatch necessary config values and override the dynamic model to use DummyModel.
    """
    device = "cpu"
    # Generate a dummy 4x4 RDM.
    # For 4 images, there are 6 unique pairs.
    rdm = np.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]], dtype=np.float32)
    # Use generate_pair_indices to extract pair information.
    from src.project.utils.pytorch_data import generate_pair_indices

    row_indices, col_indices, rdm_values = generate_pair_indices(rdm)

    # Create dummy embeddings for 4 images; each embedding has 10 features.
    embeddings = torch.tensor(np.random.rand(4, 10).astype(np.float32))

    # Override configuration values for standard training.
    monkeypatch.setattr(clip_config, "K_FOLD", False)
    config_values = {
        "TEST_SIZE": 0.5,
        "BATCH_SIZE": 2,
        "EPOCHS": 1,
        "LEARNING_RATE": 0.001,
        "HIDDEN_LAYERS": 1,
        "ACTIVATION_FUNC": "relu",
    }
    for key, value in config_values.items():
        monkeypatch.setattr(clip_config, key, value)

    # Override the dynamic model creation to return our DummyModel.
    monkeypatch.setattr(
        pytorch_training,
        "DynamicLayerSizeNeuralNetwork",
        lambda hidden_layers, activation_func=None: DummyModel(constant_value=0.5),  # noqa: ARG005
    )

    train_loss, train_acc, test_loss, test_acc = train_all(
        row_indices, col_indices, rdm_values, embeddings, device, hidden_layers=1
    )

    # Instead of asserting a fixed length, we assert that each history list is non-empty.
    # (The actual number depends on how train_test_split_pairs splits the image indices.)
    assert all(len(hist) > 0 for hist in (train_loss, train_acc, test_loss, test_acc))


def test_train_all_kfold(monkeypatch):
    """Test train_all when K_FOLD is True.

    Generate dummy pair data for 4 images so that we have enough pairs
    for the KFold split (4 images produce 6 pairs).
    Override config and the dynamic model accordingly.
    """
    device = "cpu"
    # Create a dummy 4x4 RDM.
    rdm = np.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]], dtype=np.float32)
    # Use the helper function to generate pair indices.
    from src.project.utils.pytorch_data import generate_pair_indices

    row_indices, col_indices, rdm_values = generate_pair_indices(rdm)

    # Create dummy embeddings for 4 images; each embedding has 10 features.
    embeddings = torch.tensor(np.random.rand(4, 10).astype(np.float32))

    # Set configuration for KFold mode.
    monkeypatch.setattr(clip_config, "K_FOLD", True)
    monkeypatch.setattr(clip_config, "K_FOLD_SPLITS", 2)
    monkeypatch.setattr(clip_config, "EPOCHS", 1)
    monkeypatch.setattr(clip_config, "HIDDEN_LAYERS", 1)

    # Override the dynamic model to return our DummyModel.
    monkeypatch.setattr(
        pytorch_training,
        "DynamicLayerSizeNeuralNetwork",
        lambda hidden_layers, activation_func=None: DummyModel(constant_value=0.5),  # noqa: ARG005
    )

    # Call train_all in KFold mode.
    train_loss, train_acc, test_loss, test_acc = train_all(
        row_indices, col_indices, rdm_values, embeddings, device, hidden_layers=1
    )

    # For KFold training with 2 splits, history lists should have 2 entries.
    assert all(len(hist) == 2 for hist in (train_loss, train_acc, test_loss, test_acc))
