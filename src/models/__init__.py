# flake8: noqa
import src.models as mod

from ..utils import import_submodules
from .build import MODEL_REGISTRY, build_model, BaseModel, ModelOutput
from .model import DummyModel

import_submodules(mod)
