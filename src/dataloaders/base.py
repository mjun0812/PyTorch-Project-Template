from abc import abstractmethod

from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose

from .tensor_cache import TensorCache
from .types import DatasetOutput


class BaseDataset(Dataset):
    """Abstract base class for all datasets in the framework.

    Provides a common interface for datasets with optional data transformations
    and tensor caching capabilities.

    Attributes:
        cfg: Dataset configuration dictionary.
        transforms: Torchvision transforms to apply to data.
        cache: Optional tensor cache for performance optimization.
    """

    def __init__(
        self,
        cfg: dict | None,
        transforms: Compose | None = None,
        cache: TensorCache | None = None,
    ) -> None:
        """Initialize the base dataset.

        Args:
            cfg: Dataset configuration dictionary.
            transforms: Optional data transformations.
            cache: Optional tensor cache for improved performance.
        """
        self.cfg = cfg
        self.transforms = transforms
        self.cache = cache

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> DatasetOutput:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dataset output containing the sample data.
        """
        raise NotImplementedError
