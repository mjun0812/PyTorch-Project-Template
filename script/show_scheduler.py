import matplotlib.pyplot as plt
import torch

from src.config import ConfigManager, ExperimentConfig
from src.optimizer import build_optimizer
from src.scheduler import build_lr_scheduler


@ConfigManager.argparse
def main(cfg: ExperimentConfig) -> None:
    dummy_model = torch.nn.Linear(1, 1)
    optimizer = build_optimizer(cfg.optimizer, dummy_model)
    iter_lr_scheduler, epoch_lr_scheduler = build_lr_scheduler(
        cfg.lr_scheduler, optimizer, cfg.epoch, cfg.max_iter
    )

    print(cfg.optimizer)
    print(cfg.lr_scheduler)

    x = []
    lrs = []
    for epoch in range(cfg.epoch):
        for iter_idx in range(cfg.epoch):  # dummy dataloader
            optimizer.step()
            if iter_lr_scheduler:
                iter_lr_scheduler.step(
                    epoch=(iter_idx + cfg.epoch * epoch),
                    metric=0.1 - 0.0001 * (iter_idx + cfg.epoch * epoch),
                )
                lrs.append(optimizer.param_groups[0]["lr"])
                x.append(epoch + iter_idx / cfg.epoch)
        if epoch_lr_scheduler:
            epoch_lr_scheduler.step(epoch=epoch, metric=0.1 - 0.0001 * epoch)
        lrs.append(optimizer.param_groups[0]["lr"])
        x.append(epoch + 1)

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(x, lrs, s=1)
    ax1.plot(x, lrs)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("lr")

    plt.show()


if __name__ == "__main__":
    main()
