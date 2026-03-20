from torch.utils.data import Dataset
import torch

class PairDataset(Dataset):
    def __init__(self, pair_circuits, labels):
        self.pairs = pair_circuits
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]


def collate_fn(batch):
    pairs, labels = zip(*batch)
    return list(pairs), torch.tensor(labels, dtype=torch.float32)
