# flake8: noqa
import src.evaluator as mod

from ..utils import import_submodules
from .build import build_evaluator, EVALUATOR_REGISTRY
from .base import BaseEvaluator

import_submodules(mod)
