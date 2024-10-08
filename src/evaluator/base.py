from typing import Optional

from torchmetrics import Metric


class BaseEvaluator(Metric):
    # Set to True if the metric is differentiable else set to False
    is_differentiable: Optional[bool] = None

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self) -> None:
        super().__init__()

    def update(self, targets: dict, preds: dict):
        raise NotImplementedError

    def reset(self):
        super().reset()

    def compute(self) -> dict:
        raise NotImplementedError
