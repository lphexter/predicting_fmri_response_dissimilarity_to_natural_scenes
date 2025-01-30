import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset


def generate_pair_indices(rdm):
    """Generates row_indices, col_indices for all unique pairs (i,j) where i < j,
    along with their corresponding RDM values.

    Args:
        rdm (np.ndarray): A 2D RDM matrix of shape (N, N).

    Returns:
        row_indices (np.ndarray): 1D array of row indices for each pair.
        col_indices (np.ndarray): 1D array of column indices for each pair.
        rdm_values (np.ndarray):  1D array of the RDM values for each (row, col) pair.
    """  # noqa: D205
    row_indices, col_indices = np.triu_indices(n=rdm.shape[0], k=1)
    rdm_values = rdm[row_indices, col_indices]
    return row_indices, col_indices, rdm_values


#### DEPRECATED - THIS FUNCTION CREATES OVERLAPPING PAIRS IN TRAIN/TEST DATASETS
def dep_train_test_split_pairs(row_indices, col_indices, rdm_values, test_size=0.2, random_state=42):
    all_pair_indices = np.arange(len(rdm_values))

    train_idx, test_idx, y_train, y_test = train_test_split(
        all_pair_indices, rdm_values, test_size=test_size, random_state=random_state
    )

    X_train_indices = (row_indices[train_idx], col_indices[train_idx])
    X_test_indices = (row_indices[test_idx], col_indices[test_idx])
    return X_train_indices, X_test_indices, y_train, y_test


#### Splits images into train/test first, then generates pair indices separately for each subset.
# This ensures no overlap of images between train and test.
def train_test_split_pairs(row_indices, col_indices, rdm_values, test_size=0.2, random_state=42):
    """Splits the *image indices* themselves into train and test sets, and then
    selects only the pairs belonging entirely to the train images or entirely
    to the test images. This guarantees that no image index appears in both sets.

    Args:
        row_indices (np.ndarray): 1D array of row indices for each pair.
        col_indices (np.ndarray): 1D array of column indices for each pair.
        rdm_values (np.ndarray):  1D array of RDM values for each pair.
        test_size (float):        Proportion of images used as test data (default=0.2).
        random_state (int):       Random seed for reproducibility.

    Returns:
        X_train_indices (tuple): (row_indices_train, col_indices_train).
        X_test_indices (tuple):  (row_indices_test,  col_indices_test).
        y_train (np.ndarray):    RDM values for the train pairs.
        y_test (np.ndarray):     RDM values for the test pairs.
    """  # noqa: D205
    # 1. all image indices
    all_images = np.arange(len(row_indices))

    # 2. Split those images into train/test
    train_images, test_images = train_test_split(all_images, test_size=test_size, random_state=random_state)
    train_set, test_set = set(train_images), set(test_images)

    # 3. Collect indices where both images in the pair are in the train set
    train_pair_indices = [
        i for i, (r, c) in enumerate(zip(row_indices, col_indices, strict=False)) if r in train_set and c in train_set
    ]
    # 4. Collect indices where both images in the pair are in the test set
    test_pair_indices = [
        i for i, (r, c) in enumerate(zip(row_indices, col_indices, strict=False)) if r in test_set and c in test_set
    ]

    # 5. Extract train/test pairs and their RDM values
    row_indices_train = row_indices[train_pair_indices]
    col_indices_train = col_indices[train_pair_indices]
    y_train = rdm_values[train_pair_indices]

    row_indices_test = row_indices[test_pair_indices]
    col_indices_test = col_indices[test_pair_indices]
    y_test = rdm_values[test_pair_indices]

    # 6. Return in the expected format
    X_train_indices = (row_indices_train, col_indices_train)
    X_test_indices = (row_indices_test, col_indices_test)

    return X_train_indices, X_test_indices, y_train, y_test


def prepare_data_for_kfold(row_indices, col_indices, rdm_values, loaded_features, n_splits=5):
    """Prepares data for k-fold cross-validation with non-overlapping pairs.

    Args:
        row_indices (np.ndarray): 1D array of row indices for each pair.
        col_indices (np.ndarray): 1D array of column indices for each pair.
        rdm_values (np.ndarray):  1D array of RDM values for each pair.
        loaded_features (np.ndarray): The image loaded features.
        n_splits (int):          Number of folds for cross-validation (default=5).
        is_torch (bool):         Whether to convert arrays to torch tensors (default=True).

    Returns:
        A list of (train_loader, test_loader) tuples for each fold.
    """
    # 1. all image indices
    all_images = np.arange(len(row_indices))

    # 2. Create KFold object for splitting images
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 3. Generate loaders for each fold
    loaders = []
    for train_images_index, test_images_index in kf.split(all_images):
        # Get the train and test images
        train_images = all_images[train_images_index]
        test_images = all_images[test_images_index]
        train_set, test_set = set(train_images), set(test_images)

        # Collect indices where both images in the pair are in the train set
        train_pair_indices = [
            i
            for i, (r, c) in enumerate(zip(row_indices, col_indices, strict=False))
            if r in train_set and c in train_set
        ]
        # Collect indices where both images in the pair are in the test set
        test_pair_indices = [
            i for i, (r, c) in enumerate(zip(row_indices, col_indices, strict=False)) if r in test_set and c in test_set
        ]

        # Extract train/test pairs and their RDM values
        X_train_indices = (row_indices[train_pair_indices], col_indices[train_pair_indices])
        X_test_indices = (row_indices[test_pair_indices], col_indices[test_pair_indices])
        y_train = rdm_values[train_pair_indices]
        y_test = rdm_values[test_pair_indices]

        # Create datasets
        train_dataset = PairDataset(loaded_features, X_train_indices, y_train)
        test_dataset = PairDataset(loaded_features, X_test_indices, y_test)

        # Choose an appropriate batch_size
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        loaders.append((train_loader, test_loader))

    return loaders


class PairDataset(Dataset):  # noqa: D101
    def __init__(self, embeddings, pair_indices, y_data):
        super().__init__()
        self.embeddings = embeddings
        self.row_indices, self.col_indices = pair_indices
        self.y_data = y_data

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        row_idx = self.row_indices[idx]
        col_idx = self.col_indices[idx]
        x1 = self.embeddings[row_idx]
        x2 = self.embeddings[col_idx]
        x = torch.cat([x1, x2], dim=0)
        y = torch.tensor(self.y_data[idx], dtype=torch.float32)
        return x, y