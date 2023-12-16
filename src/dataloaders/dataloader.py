import torch
from torch.utils.data import Dataset

from .build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Dataset(Dataset):
    def __init__(self, cfg, transforms=None, phase="train"):
        self.cfg = cfg
        self.transforms = transforms
        self.phase = phase
        self.images = torch.zeros((16, 3, 224, 224))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], None
