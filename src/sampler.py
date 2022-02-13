from typing import Iterator, Optional
from operator import itemgetter
from torch.utils.data import DistributedSampler

from torch.utils.data import Dataset, Sampler
import numpy as np
import torch


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class BalancedSampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = self.dataset.labels
        self.labels = torch.tensor(self.labels).to(torch.int)

        # 各ラベルの重みを設定
        label_counts = np.bincount(self.labels)
        self.label_weight = 1 - label_counts / len(dataset)
        # np.random.choiceは重みを合計1になるようにしなければならない
        self.label_weight = softmax(self.label_weight)

        # (class_num, indices)のlist
        # [[0,1,5,...],[2,3,6,...],[4,9,10,...] ...]
        self.label_index_list = []
        for i in range(len(label_counts)):
            self.label_index_list.extend(np.where(self.labels == i))

        # labelのindicesをどこまで使ったかカウントする
        self.used_indices = [0 for i in range(len(label_counts))]

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        count = 0
        while count < len(self.dataset):
            # 重みを元にlabelをランダム選択
            label_index = np.random.choice(
                len(self.label_weight), 1, p=self.label_weight
            )[0]
            # label内のindicesを選択
            index = self.used_indices[label_index]
            if len(self.label_index_list[label_index]) != 0:
                # return
                yield self.label_index_list[label_index][index]
            else:
                continue

            # label indicesの最後まで使ったら最初に戻す
            self.used_indices[label_index] += 1
            if self.used_indices[label_index] >= len(self.label_index_list[label_index]):
                self.used_indices[label_index] = 0
            count += 1


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
