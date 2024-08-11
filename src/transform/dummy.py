from .build import TRANSFORM_REGISTRY
from .transform import BaseTransform


@TRANSFORM_REGISTRY.register()
class DummyTransform(BaseTransform):
    def __call__(self, data: dict) -> dict:
        return data
