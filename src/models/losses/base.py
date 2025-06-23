from torch import nn

from ..types import LossOutput


class BaseLoss(nn.Module):
    """Abstract base class for all loss functions.

    Provides a common interface for loss computation across different
    model types and tasks.

    Attributes:
        cfg: Configuration dictionary for the loss function.
    """

    def __init__(self, cfg: dict | None) -> None:
        """Initialize the loss function.

        Args:
            cfg: Configuration dictionary containing loss-specific parameters.
        """
        super().__init__()
        self.cfg = cfg

    def forward(self, targets: dict, preds: dict) -> LossOutput:
        """Compute the loss between targets and predictions.

        Args:
            targets: Dictionary containing target values.
            preds: Dictionary containing model predictions.

        Returns:
            LossOutput containing the computed loss values.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
