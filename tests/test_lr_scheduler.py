"""LR SchedulerのLRの推移をテスト"""
from pathlib import Path

import matplotlib
import torch.nn as nn
import torch.optim as optim
import yaml
from addict import Dict

matplotlib.use("Agg")
import sys  # noqa

import matplotlib.pyplot as plt  # noqa

sys.path.append("./")
from src.scheduler import build_lr_scheduler  # noqa

EPOCH = 100
ITER = 1500


def main():
    output = Path("./doc/lr_scheduler")
    output.mkdir(parents=True, exist_ok=True)

    for path in Path("config/LR_SCHEDULER").glob("*.yaml"):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cfg = Dict(cfg)
        cfg["EPOCH"] = EPOCH
        if "MAX_LR" in cfg:
            cfg["MAX_LR"] = 0.12
        print(cfg)
        model = nn.Sequential(nn.Conv2d(3, 32, 3))
        optimizer = optim.SGD(model.parameters(), lr=0.12, momentum=0.9, weight_decay=1e-4)
        iter_scheduler, scheduler = build_lr_scheduler(cfg, optimizer, EPOCH)

        print(iter_scheduler, scheduler)

        epochs = list(range(EPOCH))
        lrs = []
        for epoch in epochs:
            for i in range(ITER):
                optimizer.step()
                if iter_scheduler:
                    iter_scheduler.step(epoch=i, metric=0.1 - 0.0001 * epoch)
                    lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step(epoch=epoch, metric=0.1 - 0.0001 * epoch)
            lrs.append(optimizer.param_groups[0]["lr"])
        epochs = list(range(len(lrs)))
        print(len(lrs))

        draw(epochs, lrs, cfg.NAME, output / f"{cfg.NAME}.png")


def draw(x, y, title, output):
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(title, fontsize=16)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(x, y, s=1)
    ax1.plot(x, y)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("lr")
    fig.savefig(output)


if __name__ == "__main__":
    main()
