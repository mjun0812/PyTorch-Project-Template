import torch.nn as nn

from .build import MODEL_REGISTRY  # noqa


@MODEL_REGISTRY.register()
class BaseModel(nn.Module):
    def __init__(self, cfg, phase="train"):
        super().__init__()
        self.cfg = cfg
        self.phase = phase

        self.loss = None
        if self.phase in ["train", "val"]:
            self.loss = self.build_loss(self.cfg)

    def build_loss(self, cfg):
        from ..losses import build_loss

        return build_loss(cfg)

    def train_forward(self, data):
        raise NotImplementedError

    def val_forward(self, data):
        raise NotImplementedError

    def test_forward(self, data):
        raise NotImplementedError

    def forward(self, data):
        if self.phase == "train":
            return self.train_forward(data)
        elif self.phase == "val":
            return self.val_forward(data)
        else:
            return self.test_forward(data)
