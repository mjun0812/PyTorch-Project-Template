from typing import Optional

from loguru import logger
from torchmetrics import MetricCollection

from ..config import ConfigManager, EvaluatorConfig, ExperimentConfig
from ..types import PhaseStr
from ..utils import Registry

EVALUATOR_REGISTRY = Registry("EVALUATOR")


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
