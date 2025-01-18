# utils/pytorch_data.py

import numpy as np
import torch
from torch.utils.data import Dataset

def generate_pair_indices(rdm):
    row_indices, col_indices = np.triu_indices(n=rdm.shape[0], k=1)
    rdm_values = rdm[row_indices, col_indices]
    return row_indices, col_indices, rdm_values

def train_test_split_pairs(row_indices, col_indices, rdm_values, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    all_pair_indices = np.arange(len(rdm_values))

    train_idx, test_idx, y_train, y_test = train_test_split(
        all_pair_indices,
        rdm_values,
        test_size=test_size,
        random_state=random_state
    )

    X_train_indices = (row_indices[train_idx], col_indices[train_idx])
    X_test_indices  = (row_indices[test_idx],  col_indices[test_idx])
    return X_train_indices, X_test_indices, y_train, y_test

class PairDataset(Dataset):
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
