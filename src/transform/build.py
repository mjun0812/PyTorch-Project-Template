from kunai import Registry
from torchvision import transforms

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
        if trans.args is None:
            transform = TRANSFORM_REGISTRY.get(trans.name)()
        else:
            params = {k: v for k, v in trans.args.items()}
            transform = TRANSFORM_REGISTRY.get(trans.name)(**params)
        transes.append(transform)
    return transforms.Compose(transes), batched_transform


# Use below Compose when using transforms has multi input.
# class Compose:
#     def __init__(self, transforms: list):
#         self.transforms = transforms

#     def __call__(self, img, mask, annotations, camera_matrix):
#         for t in self.transforms:
#             img, mask, annotations = t(img, mask, annotations)
#         return img, mask, annotations

#     def __repr__(self):
#         format_string = self.__class__.__name__ + "("
#         for t in self.transforms:
#             format_string += "\n"
#             format_string += "    {0}".format(t)
#         format_string += "\n)"
#         return format_string

# @TRANSFORM_REGISTRY.register()
# def list_to_tensor(list_obj, label):
#     """リストをtensorにキャストするだけ

#     Args:
#         list_obj (list): List Object ex. sequence
#         label (list): label

#     Returns:
#         Tensor: sequence, label
#     """
#     return torch.tensor(list_obj, dtype=torch.float64), torch.tensor(
#         label, dtype=torch.float64
#     )

# class KorniaCompose:
#     def __init__(self, cfg):
#         self.transforms = []
#         self.assing_labels = []
#         for c in cfg:
#             if c.args is None:
#                 transform = BATCHED_TRANSFORM_REGISTRY.get(c.name)()
#             else:
#                 args = OmegaConf.to_object(c.args)
#                 params = {k: v for k, v in args.items()}
#                 transform = BATCHED_TRANSFORM_REGISTRY.get(c.name)(**params)
#             self.transforms.append(transform)
#             self.assing_labels.append(c.assign_label)

#     def __call__(self, img, data):
#         # Kornia Accept image [0, 1.0]
#         img = img.float()
#         label = data["label"].float()
#         img /= 255
#         for i, t in enumerate(self.transforms):
#             img = t(img)
#             if self.assing_labels[i]:
#                 label = t(label, t._params)
#         data["label"] = label.long()
#         return img, data

#     def __repr__(self):
#         format_string = self.__class__.__name__ + "("
#         for t in self.transforms:
#             format_string += "\n"
#             format_string += "    {0}".format(t)
#         format_string += "\n)"
#         return format_string
