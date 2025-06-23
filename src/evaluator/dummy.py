from .base import BaseEvaluator
from .build import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register()
class DummyEvaluator(BaseEvaluator):
    """Dummy evaluator that returns placeholder metrics.

    This evaluator is useful for testing and development when you need
    a functioning evaluator but don't require actual metric computation.
    It returns fixed dummy values for all metrics.

    Attributes:
        prefix: Optional prefix for metric names (inherited from BaseEvaluator).
    """

    def __init__(self, prefix: str | None = None) -> None:
        """Initialize the dummy evaluator.

        Args:
            prefix: Optional prefix to add to metric names.
        """
        super().__init__(prefix)

    def update(self, targets: dict, preds: dict) -> None:
        """Update metric state (no-op for dummy evaluator).

        Args:
            targets: Dictionary containing ground truth values (ignored).
            preds: Dictionary containing model predictions (ignored).
        """
        pass

    def compute(self) -> dict:
        """Compute dummy metric values.

        Returns:
            Dictionary containing placeholder metric values for testing.
        """
        return dict(dummy_metric=0.0, another_dummy_metric=0.0)
