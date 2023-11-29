# flake8: noqa
import src.dataloaders as mod

from ..utils import import_submodules
from .build import build_dataset

import_submodules(mod)
