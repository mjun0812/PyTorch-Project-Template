import logging

import torch.optim as optim


logger = logging.getLogger()


def build_optimizer(cfg, model):
    optimizer_name = cfg.MODEL.OPTIMIZER
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.937, nesterov=True)
    logger.info(f"Using Optimizer is {optimizer_name}")
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    return scheduler
