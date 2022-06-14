import logging

import torch.optim as optim


logger = logging.getLogger()


def build_optimizer(cfg, model):
    optimizer_name = cfg.OPTIMIZER
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.937, nesterov=True)
    logger.info(f"Using Optimizer is {optimizer_name}")
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    if cfg.LR_SCHEDULER == "reduce":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    elif cfg.LR_SCHEDULER == "sgdr":
        # T_0を周期とするコサインカーブで減衰して、
        # あるところまで減衰したところで再び高いlearning rateに戻すような挙動により
        # 局所最適を脱出してもっと良いパラメータを探索します
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-5, verbose=True
        )
    logger.info(f"Using LR Scheduler is {cfg.LR_SCHEDULER}")
    return scheduler
