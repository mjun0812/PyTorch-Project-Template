from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return idx
