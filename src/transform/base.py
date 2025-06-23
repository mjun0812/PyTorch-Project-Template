from abc import abstractmethod

from torch import nn

from ..dataloaders import DatasetOutput


class BaseTransform(nn.Module):
    """Abstract base class for data transformations.

    Extends PyTorch's nn.Module to provide a standardized interface for
    data transformations that can be composed into preprocessing pipelines.
    All transformations should inherit from this class and implement the
    forward method.

    The transform operates on DatasetOutput dictionaries, allowing for
    flexible data processing across different data types and formats.
    """

    def __init__(self) -> None:
        """Initialize the base transform."""
        super().__init__()

    @abstractmethod
    def forward(self, data: DatasetOutput) -> DatasetOutput:
        """Apply the transformation to input data.

        Args:
            data: Input data dictionary from dataset.

        Returns:
            Transformed data dictionary.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return string representation of the transform.

        Returns:
            String representation showing class name and parameters.
        """
        args = [f"{k}={v}" for k, v in self.__dict__.items()]
        if len(args) == 0:
            return self.__class__.__name__ + "()"
        return self.__class__.__name__ + "(" + ", ".join(args) + ")"
