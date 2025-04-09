from .base import BaseEvaluator
from .build import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register()
class DummyEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def update(self, targets: dict, preds: dict):
        pass

    def compute(self) -> dict:
        return dict(dummy_metric=0.0, another_dummy_metric=0.0)
