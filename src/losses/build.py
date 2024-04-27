from ..utils import Registry

LOSS_REGISTRY = Registry("LOSS")


def build_loss(cfg):
    """build loss

    Args:
        cfg (OmegaConf): Hydra Conf

    Returns:
        loss: Loss function
    """
    loss_name = cfg.LOSS.NAME
    loss = LOSS_REGISTRY.get(loss_name)(cfg)

    return loss
