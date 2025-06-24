from dataclasses import dataclass

from torchvision.transforms.v2 import Compose

from .base import BaseDataset
from .build import DATASET_REGISTRY
from .tensor_cache import TensorCache
from .types import DatasetOutput


@dataclass
class DummyDatasetConfig:
    """Configuration for the dummy dataset.

    Attributes:
        path: Path to the dummy data file.
    """

    path: str = "data/dummy.csv"


@DATASET_REGISTRY.register()
class DummyDataset(BaseDataset):
    """Dummy dataset for testing and demonstration purposes.

    Generates synthetic data with fixed patterns for testing the training
    pipeline without requiring real data files.
    """

    def __init__(
        self, cfg: dict | None, transforms: Compose | None = None, cache: TensorCache | None = None
    ) -> None:
        """Initialize the dummy dataset.

        Args:
            cfg: Dataset configuration dictionary.
            transforms: Optional data transformations.
            cache: Optional tensor cache for improved performance.
        """
        super().__init__(cfg, transforms, cache)
        self.cfg = DummyDatasetConfig(**(cfg or {}))

    def __len__(self) -> int:
        """Return the number of samples in the dummy dataset.

        Returns:
            Fixed number of samples (8) for testing.
        """
        return 8

    def __getitem__(self, idx: int) -> DatasetOutput:
        """Get a dummy sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Synthetic data sample with ones tensor and modulo label.
        """
        return DatasetOutput.dummy()
