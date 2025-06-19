from abc import abstractmethod

from torchmetrics import Metric


class BaseEvaluator(Metric):
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
        super().__init__()
        self.prefix = prefix

    @abstractmethod
    def update(self, targets: dict, preds: dict) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        super().reset()

    def compute(self) -> dict:
        raise NotImplementedError
