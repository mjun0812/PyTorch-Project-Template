from .base import BaseTransform
from .build import TRANSFORM_REGISTRY


@TRANSFORM_REGISTRY.register()
class DummyTransform(BaseTransform):
    """Dummy transform that returns input data unchanged.

    This transform is useful for testing and development when you need
    a functioning transform in the pipeline but don't want to apply any
    actual data modifications. It simply passes through the input data
    without any changes.
    """

    def forward(self, data: dict) -> dict:
        """Return input data unchanged.

        Args:
            data: Input data dictionary.

        Returns:
            The same input data dictionary without modifications.
        """
        return data
