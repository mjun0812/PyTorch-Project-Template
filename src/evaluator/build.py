from torchmetrics import Metric, MetricCollection

from ..config import EvaluatorConfig
from ..utils import Registry

EVALUATOR_REGISTRY = Registry("EVALUATOR")
"""Registry for metric evaluator classes."""


def build_evaluator(cfg: list[EvaluatorConfig]) -> MetricCollection:
    """Build a collection of metric evaluators from configuration.

    Args:
        cfg: List of evaluator configurations to build.

    Returns:
        MetricCollection containing all configured evaluators.
    """
    evaluators: dict[str, Metric] = {}
    for c in cfg:
        args = c.args or {}
        evaluators[c.name] = EVALUATOR_REGISTRY.get(c.class_name)(**args)
    return MetricCollection(evaluators)
