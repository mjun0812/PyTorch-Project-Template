from .base import BaseTransform
from .build import TRANSFORM_REGISTRY


@TRANSFORM_REGISTRY.register()
class DummyTransform(BaseTransform):
    def __call__(self, data: dict) -> dict:
        return data
