import src.evaluator as mod

from ..utils import import_submodules
from .base import BaseEvaluator
from .build import EVALUATOR_REGISTRY, build_evaluator

import_submodules(mod)
