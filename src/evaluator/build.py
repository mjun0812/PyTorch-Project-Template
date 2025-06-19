from torchmetrics import Metric, MetricCollection

from ..config import EvaluatorConfig
from ..utils import Registry

EVALUATOR_REGISTRY = Registry("EVALUATOR")


def build_evaluator(cfg: list[EvaluatorConfig]) -> MetricCollection:
    evaluators: dict[str, Metric] = {}
    for c in cfg:
        args = c.args or {}
        evaluators[c.name] = EVALUATOR_REGISTRY.get(c.class_name)(**args)
    return MetricCollection(evaluators)
