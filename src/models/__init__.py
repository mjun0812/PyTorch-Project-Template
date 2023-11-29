# flake8: noqa
import src.models as mod

from ..utils import import_submodules
from .build import MODEL_REGISTRY, build_model

import_submodules(mod)
