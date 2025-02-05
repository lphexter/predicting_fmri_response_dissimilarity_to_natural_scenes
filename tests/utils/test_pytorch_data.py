import numpy as np
import torch
from src.project.utils.pytorch_data import (
    PairDataset,
    dep_train_test_split_pairs,
    generate_pair_indices,
    prepare_data_for_kfold,
    train_test_split_pairs,
)


def test_generate_pair_indices():
    """Test generate_pair_indices with a simple 3x3 RDM matrix.

    For a 3x3 matrix, there are three unique pairs:
      - (0, 1)
      - (0, 2)
      - (1, 2)
    and the corresponding RDM values should be extracted correctly.
    """
    # Define a simple symmetric RDM.
    rdm = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    row_indices, col_indices, rdm_values = generate_pair_indices(rdm)
    # Expected pairs: (0,1), (0,2), (1,2)
    expected_row = np.array([0, 0, 1])
    expected_col = np.array([1, 2, 2])
    expected_values = np.array([1, 2, 3])
    np.testing.assert_array_equal(row_indices, expected_row)
    np.testing.assert_array_equal(col_indices, expected_col)
    np.testing.assert_array_equal(rdm_values, expected_values)


def test_dep_train_test_split_pairs():
    """Test dep_train_test_split_pairs (deprecated version) with dummy pair indices.

    The function splits the pair indices (based on a fixed random state) so that
    the union of training and testing pairs recovers all the original pairs.
    """
    # Dummy pair data (as might be returned from generate_pair_indices)
    row_indices = np.array([0, 0, 1])
    col_indices = np.array([1, 2, 2])
    rdm_values = np.array([1, 2, 3])
    # Use a test_size of 0.33 (approximately one-third of the pairs) and fixed seed.
    X_train, X_test, y_train, y_test = dep_train_test_split_pairs(
        row_indices, col_indices, rdm_values, test_size=0.33, random_state=42
    )
    # Check that the total number of pairs equals the original count.
    total_pairs = len(y_train) + len(y_test)
    assert total_pairs == len(rdm_values)
    # Verify that the combined RDM values (sorted) match the original ones.
    np.testing.assert_array_equal(np.sort(np.concatenate([y_train, y_test])), np.sort(rdm_values))


def test_train_test_split_pairs():
    """Test train_test_split_pairs with dummy pair data.

    This function splits image indices (from which the pair indices were computed)
    into training and testing groups and then selects only those pairs where both images
    belong to the same split.

    We simply verify that the returned RDM values come from the original array and that
    the total number of returned pairs does not exceed the original number.
    """
    # Create dummy pair indices for a hypothetical 4x4 RDM.
    # (For a 4x4 matrix, one common way is to have:
    #  pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3))
    # Here we simulate a scenario with only a subset.
    row_indices = np.array([0, 0, 0, 1, 1, 2])
    col_indices = np.array([1, 2, 3, 2, 3, 3])
    rdm_values = np.array([10, 20, 30, 40, 50, 60])
    X_train, X_test, y_train, y_test = train_test_split_pairs(
        row_indices, col_indices, rdm_values, test_size=0.5, random_state=42
    )
    # Since the splitting is done over the indices (as defined by len(row_indices)),
    # some pairs may be dropped. Check that the total returned pairs do not exceed the original.
    total_pairs_returned = len(y_train) + len(y_test)
    assert total_pairs_returned <= len(rdm_values)
    # Verify that every returned RDM value is present in the original array.
    combined = np.concatenate([y_train, y_test])
    for val in combined:
        assert val in rdm_values


def test_prepare_data_for_kfold():
    """Test prepare_data_for_kfold by simulating a small dataset.

    We create dummy pair indices and RDM values for 4 images (hence a few pairs)
    and dummy loaded features for 4 images. We then verify that the function returns
    a list of (train_loader, test_loader) tuples and that each loader produces batches
    with the expected dimensions.
    """
    # Dummy pair indices (assumed to come from a 4x4 matrix; here we provide a small set).
    row_indices = np.array([0, 0, 1, 1])
    col_indices = np.array([1, 2, 2, 3])
    rdm_values = np.array([0.1, 0.2, 0.3, 0.4])
    # Create dummy loaded features for 4 images, each with 10 features.
    loaded_features = torch.tensor(np.random.rand(4, 10).astype(np.float32))
    # Use 2 folds for simplicity.
    loaders = prepare_data_for_kfold(row_indices, col_indices, rdm_values, loaded_features, n_splits=2)
    # Check that we obtained 2 folds.
    assert len(loaders) == 2
    # For each fold, verify that the data loaders yield batches with the correct feature dimensions.
    for train_loader, test_loader in loaders:
        # Get one batch from train_loader, if available.
        for batch in train_loader:
            x, y = batch
            # The dataset concatenates two feature vectors (each of length 10) → expected shape: (batch_size, 20)
            assert x.shape[1] == 20
            # y should be a 1D tensor (one value per pair).
            assert len(y.shape) == 1
            break  # Only one batch is needed for this check.
        for batch in test_loader:
            x, y = batch
            assert x.shape[1] == 20
            assert len(y.shape) == 1
            break


def test_pair_dataset():
    """Test the PairDataset

    Verify that it correctly concatenates the embeddings
    for a given pair of image indices and returns the expected label.
    """
    # Create dummy embeddings for 3 images; each embedding has 5 features.
    embeddings = torch.tensor(np.array([np.arange(5), np.arange(5, 10), np.arange(10, 15)], dtype=np.float32))
    # Define dummy pair indices: for instance, pairs (0,1) and (1,2).
    row_indices = np.array([0, 1])
    col_indices = np.array([1, 2])
    y_data = np.array([0.5, 0.8])
    dataset = PairDataset(embeddings, (row_indices, col_indices), y_data)
    # Check that the length of the dataset equals the number of pairs.
    assert len(dataset) == 2

    # Test the first item.
    x, y = dataset[0]
    # Expected: concatenation of embeddings[0] and embeddings[1] → tensor of shape (10,)
    expected = torch.cat([torch.tensor(embeddings[0]), torch.tensor(embeddings[1])], dim=0)
    torch.testing.assert_allclose(x, expected)
    # Verify that the label is a float tensor with the correct value.
    assert y.dtype == torch.float32
    torch.testing.assert_allclose(y, torch.tensor(0.5, dtype=torch.float32))
