from .base import BaseEvaluator
from .build import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register()
class DummyEvaluator(BaseEvaluator):
    def __init__(self, prefix: str | None = None) -> None:
        super().__init__(prefix)

    def update(self, targets: dict, preds: dict) -> None:
        pass

    def compute(self) -> dict:
        return dict(dummy_metric=0.0, another_dummy_metric=0.0)
