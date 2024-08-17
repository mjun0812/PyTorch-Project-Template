# flake8: noqa
import src.models as mod

from ..utils import import_submodules
from .build import MODEL_REGISTRY, build_model
from .base import BaseModel
from .model import DummyModel
from .backbone import build_backbone, BackboneConfig

import_submodules(mod)
