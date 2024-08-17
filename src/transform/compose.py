from omegaconf import OmegaConf

from ..config import TransformConfig
from .base import BaseTransform


# Use below Compose when using transforms has multi input.
class MultiCompose:
    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def __call__(self, img, data):
        for t in self.transforms:
            img, data = t(img, data)
        return img, data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class KorniaCompose:
    def __init__(self, cfg: list[TransformConfig]):
        from .build import BATCHED_TRANSFORM_REGISTRY

        self.transforms = []
        self.assing_labels = []

        for c in cfg:
            if c.args is None:
                transform = BATCHED_TRANSFORM_REGISTRY.get(c.name)()
            else:
                args = OmegaConf.to_object(c.args)
                transform = BATCHED_TRANSFORM_REGISTRY.get(c.name)(**args)
            self.transforms.append(transform)

            if hasattr(transform, "assign_label"):
                self.assing_labels.append(transform.assign_label)

    def __call__(self, data):
        # Kornia Accept image [0, 1.0]
        img = data["image"].float()
        img /= 255
        label = data["label"].float()
        for i, t in enumerate(self.transforms):
            img = t(img)
            if self.assing_labels[i]:
                label = t(label, t._params)
        data["image"] = img
        data["label"] = label.long()
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
