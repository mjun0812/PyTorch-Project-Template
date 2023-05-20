from pathlib import Path
from omegaconf import OmegaConf

import torch.optim as optim
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

import sys  # noqa

sys.path.append("./")
from src.scheduler import build_lr_scheduler  # noqa

EPOCH = 100


def main():
    output = Path("./doc/lr_scheduler")
    output.mkdir(parents=True, exist_ok=True)

    for path in Path("config/LR_SCHEDULER").glob("*.yaml"):
        cfg = OmegaConf.load(path)
        cfg["EPOCH"] = EPOCH
        if "MAX_LR" in cfg:
            print("aaaa")
            cfg["MAX_LR"] = 1e-3
        print(cfg)
        model = nn.Sequential(nn.Conv2d(3, 32, 3))
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.1)
        scheduler = build_lr_scheduler(cfg, optimizer, EPOCH)

        epochs = list(range(100))
        lrs = []
        for epoch in epochs:
            optimizer.zero_grad()
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step(epoch=epoch, metric=0.1 - 0.0001 * epoch)
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
