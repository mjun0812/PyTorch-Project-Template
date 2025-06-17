from torchmetrics import Metric, MetricCollection

from ..config import ConfigManager, EvaluatorConfig
from ..utils import Registry

EVALUATOR_REGISTRY = Registry("EVALUATOR")


def build_evaluator(cfg: list[EvaluatorConfig]) -> MetricCollection:
    evaluators: dict[str, Metric] = {}
    for c in cfg:
        args = ConfigManager.to_object(c.args.copy()) if c.args is not None else {}
        evaluators[c.name] = EVALUATOR_REGISTRY.get(c.class_name)(**args)
    return MetricCollection(evaluators)
