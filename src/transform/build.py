from omegaconf import OmegaConf

from ..utils import Registry

USE_V2 = False
try:
    USE_V2 = True
    import torchvision.transforms.v2 as T
except ImportError:
    print("ImportError: torchvision.transforms.v2")
    from torchvision import transforms as T

TRANSFORM_REGISTRY = Registry("TRANSFORM")
BATCHED_TRANSFORM_REGISTRY = Registry("BATCHED_TRANSFORM")


def build_transforms(cfg, phase="train"):
    cfg = cfg.get(f"{phase.upper()}_DATASET").TRANSFORMS

    batched_transform = None
    if phase == "train":
        if "TRAIN_BATCH" in cfg:
            batched_transform = None
        cfg = cfg.TRAIN
    elif phase == "val":
        cfg = cfg.VAL
    elif phase == "test":
        cfg = cfg.TEST

    transes = []
    for trans in cfg:
        transes.append(get_transform(trans))
    return T.Compose(transes), batched_transform


def get_transform(cfg):
    if cfg["args"] is None:
        transform = TRANSFORM_REGISTRY.get(cfg["name"])()
    else:
        params = cfg["args"]
        transform = TRANSFORM_REGISTRY.get(cfg["name"])(**params)
    return transform


# Use below Compose when using transforms has multi input.
class MultiCompose:
    def __init__(self, transforms: list):
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
    def __init__(self, cfg):
        self.transforms = []
        self.assing_labels = []
        for c in cfg:
            if c.args is None:
                transform = BATCHED_TRANSFORM_REGISTRY.get(c.name)()
            else:
                args = OmegaConf.to_object(c.args)
                params = {k: v for k, v in args.items()}
                transform = BATCHED_TRANSFORM_REGISTRY.get(c.name)(**params)
            self.transforms.append(transform)
            self.assing_labels.append(c.assign_label)

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
