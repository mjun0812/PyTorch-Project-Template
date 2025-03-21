import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.config import ConfigManager, ExperimentConfig
from src.optimizer import build_optimizer
from src.scheduler import build_lr_scheduler

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True


@ConfigManager.argparse
def main(cfg: ExperimentConfig):
    dummy_model = torch.nn.Linear(1, 1)
    optimizer = build_optimizer(cfg, dummy_model)
    iter_lr_scheduler, epoch_lr_scheduler = build_lr_scheduler(cfg, optimizer)

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

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # OpenCVでグラフを表示
    cv2.imshow("Learning Rate Schedule", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
