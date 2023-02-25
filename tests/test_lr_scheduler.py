import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


def main():
    model = nn.Sequential(nn.Conv2d(3, 32, 3))
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.1)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=95, T_mult=5, eta_min=1e-5
    )

    epochs = list(range(100))
    lrs = []
    for _ in epochs:
        lrs.append(lr_scheduler.get_last_lr()[0])
        lr_scheduler.step()

    draw(epochs, lrs, "")


def draw(x, y, title):
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(title, fontsize=16)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(x, y, s=1)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("lr")
    plt.show()


if __name__ == "__main__":
    main()
