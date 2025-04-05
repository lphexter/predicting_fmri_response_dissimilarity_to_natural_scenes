import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch import float32, tensor
from torch.utils.data import DataLoader, Dataset

from ..config import clip_config

MID_VALUE_MIN = 0.85
MID_VALUE_MAX = 1.15


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
    # 1. Create array of image indices for calculating pairs
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
        n_splits (int):          Number of folds for cross-validation.
        is_torch (bool):         Whether to convert arrays to torch tensors (default=True).

    Returns:
        A list of (train_loader, test_loader) tuples for each fold.
    """
    # 1. Create array of image indices for calculating pairs
    all_images = np.arange(len(row_indices))

    # 2. Create KFold object for splitting images
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 3. Generate loaders for each fold
    loaders = []
    for train_images_index, test_images_index in kf.split(all_images):
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

        train_dataset = PairDataset(loaded_features, X_train_indices, y_train)
        test_dataset = PairDataset(loaded_features, X_test_indices, y_test)

        train_loader = DataLoader(train_dataset, batch_size=clip_config.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=clip_config.BATCH_SIZE, shuffle=False)

        loaders.append((train_loader, test_loader))

    return loaders


#########################################
#    DATA HELPER METHODS - NEWER MODELS
#########################################


def get_balanced_pairs_list(rdm, n_extremes=50):
    """Get a balanced distribution across the data.

    Returns a list of pairs (i, j) where j includes extreme pairs
    (lowest/highest) plus mid-range values centered around 1.0.

    Parameters:
        rdm (np.ndarray): 2D representational dissimilarity matrix.
        n_extremes (int): Number of lowest and highest pairs to select per image.

    Returns:
        List[Tuple[int, int]]: List of tuples representing image index pairs.
    """
    n_images = rdm.shape[0]
    pairs_list = []

    # Get extreme pairs using the existing function
    extreme_pairs_list = get_extreme_pairs_list(rdm, n_extremes)
    # Create a dictionary to store extreme pairs for each image
    extreme_pairs_dict = {}
    for i, j in extreme_pairs_list:
        extreme_pairs_dict.setdefault(i, []).append(j)

    for i in range(n_images):
        extreme_pairs = extreme_pairs_dict.get(i, [])

        # Find mid-range values (centered around 1.0)
        valid_indices = np.delete(np.arange(n_images), i)  # Exclude self-comparison
        values = rdm[i, valid_indices]
        sorted_order = np.argsort(values)
        sorted_indices = valid_indices[sorted_order]

        mid_mask = (values >= MID_VALUE_MIN) & (
            values <= MID_VALUE_MAX
        )  # Peak around 1.0, Curtomise value range until getting even distribution
        mid_indices = sorted_indices[mid_mask]

        if len(mid_indices) > n_extremes:
            mid_indices = np.random.choice(
                mid_indices, int(n_extremes // 0.3), replace=False
            )  # Customise //number until getting even distribution

        # Combine extreme and mid-range pairs
        combined = np.concatenate([extreme_pairs, mid_indices])

        # Append tuple pairs for this image
        for j in combined:
            pairs_list.append((int(i), int(j)))  # noqa: PERF401

    return pairs_list


def get_extreme_pairs_list(rdm, n_extremes=50):
    """Get a distribution of only extreme values in the data.

    Returns a list of pairs (i, j) where j is one of the most extreme pairs (lowest/highest)
    for image i, without applying any lower or upper bound.

    Parameters:
        rdm (np.ndarray): 2D representational dissimilarity matrix.
        n_extremes (int): Number of lowest and highest pairs to select per image.

    Returns:
        List[Tuple[int, int]]: List of tuples representing image index pairs.
    """
    n_images = rdm.shape[0]
    pairs_list = []

    for i in range(n_images):
        # Exclude self-comparison
        valid_indices = np.delete(np.arange(n_images), i)  # All indices except i

        values = rdm[i, valid_indices]
        sorted_order = np.argsort(values)  # Sort in increasing order
        sorted_indices = valid_indices[sorted_order]

        # Select n_extremes lowest and highest, ensuring we don't exceed available data
        low_indices = sorted_indices[:n_extremes]
        high_indices = sorted_indices[-n_extremes:] if len(sorted_indices) >= n_extremes else sorted_indices

        combined = np.concatenate([low_indices, high_indices])

        # Append tuple pairs for this image
        for j in combined:
            pairs_list.append((int(i), int(j)))  # noqa: PERF401

    return pairs_list


def get_valid_pair_indices(rdm, metric="correlation", distribution_type="all"):
    # default - return all pairs of indices
    n_images = rdm.shape[0]
    if distribution_type in ("all", "colors"):
        valid_pair_indices = [(i, j) for i, j in zip(*np.triu_indices(n_images, k=1), strict=False)]
    elif distribution_type == "extremes":
        valid_pair_indices = get_extreme_pairs_list(rdm)
    elif distribution_type == "balanced":
        if metric == "euclidean":
            raise ValueError("Can't use balanced distribution for euclidean metric")
        valid_pair_indices = get_balanced_pairs_list(rdm)
    else:
        raise ValueError("Invalid distribution type")

    return valid_pair_indices


def make_pairs(rdm_binary, rdm_numeric, indices):
    """Build pairwise data from a precomputed list of pairs.

    Args:
        rdm_binary (np.ndarray): shape (N, N), binary similarity/dissimilarity matrix.
        rdm_numeric (np.ndarray): shape (N, N), numeric dissimilarity matrix.
        indices (list): Either a list of integers or a list of tuple pairs.

    Returns:
        X_pairs (np.ndarray): shape (num_pairs, 1024), indices of embeddings for each pair.
        y_pairs (np.ndarray): shape (num_pairs,), binary labels.
        y_numeric_pairs (np.ndarray): shape (num_pairs,), numeric dissimilarity values.
    """
    pair_indices = np.array(indices)  # Convert to NumPy array for indexing
    X_data = np.stack((pair_indices[:, 0], pair_indices[:, 1]), axis=-1)  # Store pair indices instead of embeddings
    y_binary = rdm_binary[pair_indices[:, 0], pair_indices[:, 1]]
    y_numeric = rdm_numeric[pair_indices[:, 0], pair_indices[:, 1]]

    return X_data, y_binary, y_numeric


def get_train_and_test_pairs(train_indices, test_indices, shuffled_indices):
    # Convert to sets for membership testing.
    train_img_set = set(train_indices)
    test_img_set = set(test_indices)

    # Filter pairs so that only pairs with both images in the train (or test) set are used.
    train_pairs = [pair for pair in shuffled_indices if pair[0] in train_img_set and pair[1] in train_img_set]
    test_pairs = [pair for pair in shuffled_indices if pair[0] in test_img_set and pair[1] in test_img_set]

    return train_pairs, test_pairs


# Data class for SVM
class PairedData:
    """A dataset class that holds embeddings and their associated pairing indices.

    Each data point is constructed by concatenating a pair of embeddings.

    Attributes:
        embeddings (np.ndarray): An array of embeddings.
        pair_indices (list of tuple): A list of index pairs indicating which embeddings to pair.
    """

    def __init__(self, embeddings, pair_indices):
        """Initialize the PairedData instance.

        Args:
            embeddings (np.ndarray): An array containing the embeddings.
            pair_indices (list of tuple): A list where each tuple contains two indices
                                          that specify a pair of embeddings.
        """
        self.embeddings = embeddings
        self.pair_indices = pair_indices

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        idx1, idx2 = self.pair_indices[idx]
        emb1 = self.embeddings[idx1]
        emb2 = self.embeddings[idx2]
        return np.concatenate([emb1, emb2])


class PairedDataset(Dataset):
    """A PyTorch Dataset for contrastive learning

    Provides pairs of embeddings and corresponding labels.
    The embeddings and labels are converted to PyTorch tensors for model consumption.

    Attributes:
        paired_embeddings (torch.Tensor): Tensor of embeddings.
        paired_indices (list of tuple): A list of index pairs specifying which embeddings to pair.
        labels (torch.Tensor): Tensor of labels corresponding to each pair.
    """

    def __init__(self, paired_embeddings, paired_indices, labels):
        """Initialize the PairedDataset instance.

        Args:
            paired_embeddings (array-like): The embeddings to be paired.
            paired_indices (list of tuple): A list of tuples, where each tuple contains two indices
                                            specifying which embeddings to pair.
            labels (array-like): The labels for each pair.
        """
        self.paired_embeddings = tensor(paired_embeddings, dtype=float32)
        self.paired_indices = paired_indices
        self.labels = tensor(labels, dtype=float32)

    def __len__(self):
        return len(self.paired_indices)

    def __getitem__(self, idx):
        idx1, idx2 = self.paired_indices[idx]
        emb1 = self.paired_embeddings[idx1]
        emb2 = self.paired_embeddings[idx2]
        label = self.labels[idx]
        return emb1, emb2, label


# Old MLP Pair Dataset Class
class PairDataset(Dataset):
    """A PyTorch dataset for handling paired embeddings.

    This dataset is designed to store pairs of embeddings along with their corresponding labels.
    It takes a set of embeddings, indices defining pairs, and target values.

    Attributes:
        embeddings (torch.Tensor): The tensor containing all embeddings.
        row_indices (numpy.ndarray): The indices of the first item in each pair.
        col_indices (numpy.ndarray): The indices of the second item in each pair.
        y_data (numpy.ndarray): The target values associated with each pair.

    Methods:
        __len__(): Returns the number of pairs in the dataset.
        __getitem__(idx): Retrieves the paired embeddings and target value for a given index.
    """

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
