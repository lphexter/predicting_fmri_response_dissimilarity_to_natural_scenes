import numpy as np
import pytest
import torch

from project.utils.pytorch_data import (
    PairDataset,
    PairedData,
    PairedDataset,
    dep_train_test_split_pairs,
    generate_pair_indices,
    get_balanced_pairs_list,
    get_extreme_pairs_list,
    get_train_and_test_pairs,
    get_valid_pair_indices,
    make_pairs,
    prepare_data_for_kfold,
    train_test_split_pairs,
)

#########################################
# Tests for new pair/index helper functions
#########################################


def test_get_extreme_pairs_list():
    """Test get_extreme_pairs_list with a small 3x3 RDM and n_extremes=1."""
    rdm = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    pairs = get_extreme_pairs_list(rdm, n_extremes=1)
    # For each image, expect two pairs: one with the lowest and one with the highest value.
    # For image 0: valid indices [1,2] → expected pairs (0,1) and (0,2)
    # For image 1: valid indices [0,2] → expected pairs (1,0) and (1,2)
    # For image 2: valid indices [0,1] → expected pairs (2,0) and (2,1)
    expected = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    assert sorted(pairs) == sorted(expected), f"Expected {expected}, got {pairs}"


def test_get_balanced_pairs_list():
    """Test get_balanced_pairs_list with an RDM that provides sufficient mid-range values.

    Constructs an RDM for 5 images where all off-diagonals are 1.0 (which lies within the mid-range [0.85, 1.15]).
    Then, verifies that each returned pair is within valid indices and that at least one pair is returned.
    """
    n_images = 5
    # Create an RDM where all off-diagonals are 1.0 (mid-range) and the diagonal is 0.
    rdm = np.ones((n_images, n_images))
    np.fill_diagonal(rdm, 0)

    # n_extremes is 1, so the mid-range sampling will attempt to sample int(1 // 0.3)=3 elements.
    pairs = get_balanced_pairs_list(rdm, n_extremes=1)

    for i, j in pairs:
        assert 0 <= i < n_images, f"Pair index i={i} out of bounds."
        assert 0 <= j < n_images, f"Pair index j={j} out of bounds."
    assert len(pairs) > 0, "No pairs returned from get_balanced_pairs_list."


def test_get_valid_pair_indices_all():
    """Test get_valid_pair_indices with distribution_type 'all' returns all unique pairs."""
    n = 4
    # Create a dummy symmetric RDM (values don't matter here).
    rdm = np.zeros((n, n))
    valid_pairs = get_valid_pair_indices(rdm, distribution_type="all")
    # Expected all unique pairs (i,j) for i < j
    expected = [(i, j) for i in range(n) for j in range(i + 1, n)]
    assert sorted(valid_pairs) == sorted(expected)


def test_get_valid_pair_indices_extremes():
    """Test get_valid_pair_indices with distribution_type 'extremes' returns the same as get_extreme_pairs_list."""
    rdm = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    valid_pairs = get_valid_pair_indices(rdm, distribution_type="extremes")
    expected = get_extreme_pairs_list(rdm)  # default n_extremes=50, but our small RDM is fine
    assert sorted(valid_pairs) == sorted(expected)


def test_get_valid_pair_indices_balanced_euclidean():
    """Test that get_valid_pair_indices with distribution_type 'balanced' and euclidean metric raises ValueError."""
    rdm = np.eye(4)
    with pytest.raises(ValueError):  # noqa: PT011
        get_valid_pair_indices(rdm, metric="euclidean", distribution_type="balanced")


def test_make_pairs():
    """Test make_pairs returns arrays with expected shapes and values."""
    # Create a dummy RDM (binary and numeric) for 4 images.
    rdm_binary = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
    rdm_numeric = np.array([[0, 0.2, 0.3, 0.4], [0.2, 0, 0.5, 0.6], [0.3, 0.5, 0, 0.7], [0.4, 0.6, 0.7, 0]])
    # Define indices as list of pairs.
    indices = [(0, 1), (2, 3)]
    X_data, y_binary, y_numeric = make_pairs(rdm_binary, rdm_numeric, indices)
    # Expect X_data shape (2,2), y_binary shape (2,), y_numeric shape (2,)
    assert X_data.shape == (2, 2)
    assert y_binary.shape == (2,)
    assert y_numeric.shape == (2,)
    # Verify values
    np.testing.assert_array_equal(X_data, np.array([[0, 1], [2, 3]]))
    np.testing.assert_array_equal(y_binary, np.array([1, 1]))
    np.testing.assert_array_equal(y_numeric, np.array([0.2, 0.7]))


def test_get_train_and_test_pairs():
    """Test get_train_and_test_pairs separates pairs based on provided train/test image indices."""
    # Suppose we have 5 images.
    train_indices = [0, 1, 2]
    test_indices = [3, 4]
    # Shuffled indices: list of pairs (as tuples)
    shuffled_indices = [(0, 1), (0, 3), (1, 2), (3, 4), (2, 4)]
    train_pairs, test_pairs = get_train_and_test_pairs(train_indices, test_indices, shuffled_indices)
    # Expected: train pairs are those with both indices in {0,1,2} and test pairs with both in {3,4}.
    expected_train = [(0, 1), (1, 2)]
    expected_test = [(3, 4)]
    assert sorted(train_pairs) == sorted(expected_train)
    assert sorted(test_pairs) == sorted(expected_test)


#########################################
# Tests for new Dataset classes
#########################################


def test_paired_data():
    """Test the PairedData class to verify correct concatenation of embeddings."""
    # Create dummy embeddings: 5 images, each with 4 features.
    embeddings = np.arange(20).reshape(5, 4)
    pair_indices = [(0, 1), (2, 3)]
    dataset = PairedData(embeddings, pair_indices)
    # Length should equal number of pairs.
    assert len(dataset) == 2
    # Test first item: expect concatenation of embeddings[0] and embeddings[1]
    item = dataset[0]
    expected = np.concatenate([embeddings[0], embeddings[1]])
    np.testing.assert_array_equal(item, expected)


def test_paired_dataset():
    """Test the PairedDataset class to verify proper tensor conversion and pair retrieval."""
    # Create dummy embeddings: 5 images, each with 3 features.
    embeddings = np.arange(15).reshape(5, 3).astype(np.float32)
    pair_indices = [(0, 2), (1, 3)]
    labels = np.array([0.5, 0.8], dtype=np.float32)
    dataset = PairedDataset(embeddings, pair_indices, labels)
    # Check length.
    assert len(dataset) == 2
    # Get first item.
    emb1, emb2, label = dataset[0]
    # Expected: emb1 = embeddings[0], emb2 = embeddings[2]
    expected_emb1 = torch.tensor(embeddings[0])
    expected_emb2 = torch.tensor(embeddings[2])
    torch.testing.assert_allclose(emb1, expected_emb1)
    torch.testing.assert_allclose(emb2, expected_emb2)
    # Label should match.
    torch.testing.assert_allclose(label, torch.tensor(0.5, dtype=torch.float32))


#########################################
# Tests for old pytorch data functions
#########################################


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
