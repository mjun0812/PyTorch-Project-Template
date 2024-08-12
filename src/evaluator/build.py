from typing import Optional

from loguru import logger
from torchmetrics import Metric, MetricCollection

from ..alias import PhaseStr
from ..config import ConfigManager, EvaluatorConfig, ExperimentConfig
from ..utils import Registry

EVALUATOR_REGISTRY = Registry("EVALUATOR")


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


def build_evaluator(cfg: ExperimentConfig, phase: PhaseStr = "train") -> Optional[MetricCollection]:
    cfg_eval: Optional[list[EvaluatorConfig]] = cfg.evaluator.get(phase)

    if cfg_eval is None or len(cfg_eval) == 0:
        return None

    evaluators = []
    for c in cfg_eval:
        args = ConfigManager.to_object(c.args.copy()) if c.args is not None else {}
        evaluators.append(EVALUATOR_REGISTRY.get(c.name)(**args))
    logger.info(f"{phase.capitalize()} Evaluators: {evaluators}")
    return MetricCollection(evaluators)
