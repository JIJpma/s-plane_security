import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

from DU_model.Transformer.t_size import TransformerNN


class PTPDataset(Dataset):
    def __init__(self, data: pd.DataFrame, labels: pd.Series, seq_len: int):
        self.data = data.values.astype(np.float32)
        self.labels = labels.values.astype(np.int64)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        window = torch.tensor(self.data[idx : idx + self.seq_len])
        label = int(self.labels[idx : idx + self.seq_len].any())
        return window, label


def load_model(
    path: str, device: torch.device, *, slice_len: int = 64, nhead: int = 3
) -> TransformerNN:
    model = TransformerNN(slice_len=slice_len, nhead=nhead)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model: TransformerNN, loader: DataLoader, device: torch.device):
    all_probs, all_labels = [], []
    for inputs, labels in loader:
        probs = model(inputs.to(device)).cpu().squeeze(-1)
        all_probs.append(probs)
        all_labels.append(labels)
    return torch.cat(all_probs), torch.cat(all_labels)
