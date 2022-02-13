from kunai import Registry

MODEL_REGISTRY = Registry("MODEL")


def build_model(cfg):
    """build model

    Args:
        cfg (OmegaConf): Hydra Conf

    Returns:
        model: Torch.model
    """
    model_name = cfg.MODEL.MODEL
    model = MODEL_REGISTRY.get(model_name)(cfg)

    return model
