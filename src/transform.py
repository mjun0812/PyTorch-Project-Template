from torchvision import transforms

from .utils.registry import Registry

TRANSFORM_REGISTRY = Registry("TRANSFORM")


def build_transforms(cfg, phase="train"):
    cfg = cfg.DATASET.TRANSFORMS
    if phase == "train":
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
    return transforms.Compose(transes)


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
