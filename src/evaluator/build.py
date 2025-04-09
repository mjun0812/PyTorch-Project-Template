from torchmetrics import MetricCollection

from ..config import ConfigManager, EvaluatorConfig
from ..utils import Registry

EVALUATOR_REGISTRY = Registry("EVALUATOR")


def build_evaluator(cfg: list[EvaluatorConfig]) -> MetricCollection:
    evaluators = []
    for c in cfg:
        args = ConfigManager.to_object(c.args.copy()) if c.args is not None else {}
        evaluators.append(EVALUATOR_REGISTRY.get(c.name)(**args))
    return MetricCollection(evaluators)
