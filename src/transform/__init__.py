# flake8: noqa
import src.transform as mod

from ..utils import import_submodules
from .build import TRANSFORM_REGISTRY, build_transforms

import_submodules(mod)
