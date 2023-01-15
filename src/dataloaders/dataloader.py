from torch.utils.data import Dataset
from .build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Dataset(Dataset):
    def __init__(self, cfg, transforms=None, phase="train"):
        self.cfg = cfg
        self.transforms = transforms
        self.phase = phase

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return idx
