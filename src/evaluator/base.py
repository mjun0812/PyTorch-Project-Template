from abc import abstractmethod

from torchmetrics import Metric


class BaseEvaluator(Metric):
    """Abstract base class for custom metric evaluators.

    Extends TorchMetrics' Metric class to provide a standardized interface
    for evaluation metrics in the framework.

    Attributes:
        is_differentiable: Whether the metric is differentiable.
        higher_is_better: Whether higher values indicate better performance.
        full_state_update: Whether metric requires global state for updates.
        prefix: Optional prefix for metric names.
    """

    # Set to True if the metric is differentiable else set to False
    is_differentiable: bool | None = None

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool | None = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self, prefix: str | None = None) -> None:
        """Initialize the base evaluator.

        Args:
            prefix: Optional prefix to add to metric names.
        """
        super().__init__()
        self.prefix = prefix

    @abstractmethod
    def update(self, targets: dict, preds: dict) -> None:
        """Update the metric state with new predictions and targets.

        Args:
            targets: Dictionary containing ground truth values.
            preds: Dictionary containing model predictions.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the metric state."""
        super().reset()

    def compute(self) -> dict:
        """Compute the final metric value.

        Returns:
            Dictionary containing computed metric values.
        """
        raise NotImplementedError
