import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torch import optim

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.config import ConfigManager, ExperimentConfig

matplotlib.use("Agg")
# 論文用にFontを変更する
font_manager.fontManager.addfont("./etc/Times_New_Roman.ttf")
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 18})


def test_train_test_resume():
    common_options = [
        "mlflow.use=false",
        "output=/tmp/test",
        "epoch=2",
        "val_interval=1",
        "use_ram_cache=false",
    ]
    if torch.cuda.is_available():
        common_options.append("use.gpu=0")
    else:
        common_options.append("use_cpu=true")
    print("start training test")
    process = subprocess.run(
        ["python", "train.py", "config/dummy.yaml"] + common_options,
        stdout=subprocess.PIPE,
    )
    assert process.returncode == 0
    output = process.stdout.decode("utf-8")
    print("end training test")

    output_cmd = None
    output_rows = output.split("\n")
    output_rows.reverse()
    for line in output_rows:
        if "test cmd:" in line:
            output_cmd = line.split("test cmd: ")[1].strip()
            break
    assert output_cmd is not None

    print("start test test")
    print("test cmd:", output_cmd)
    process = subprocess.run(
        output_cmd.split(" "),
        stdout=subprocess.PIPE,
    )
    assert process.returncode == 0
    print("end test test")

    print("start resume test")
    cmd = output_cmd.replace("test.py", "train.py").split(" ")
    process = subprocess.run(
        cmd + ["epoch=3"],
        stdout=subprocess.PIPE,
    )
    assert process.returncode == 0
    print("end resume test")


def _load_config(path: str) -> ExperimentConfig:
    # From Dataclass
    cfg = OmegaConf.structured(ExperimentConfig)
    # From File
    cfg_from_file = ConfigManager.build_config_from_file(path)
    cfg = ConfigManager.merge(cfg, cfg_from_file)
    return cfg


def test_config():
    cfg = _load_config(os.path.join(os.path.dirname(__file__), "../config/dummy.yaml"))
    print(ConfigManager.pretty_text(cfg))


def test_dataloader():
    from src.config import DatasetsConfig
    from src.dataloaders import build_dataloader, build_dataset, build_sampler
    from src.transform import build_batched_transform, build_transforms

    config_dir = os.path.join(os.path.dirname(__file__), "../config/__base__/dataset")
    config_dir = Path(config_dir)
    for config_path in config_dir.glob("*.yaml"):
        dataset_cfg = OmegaConf.load(config_path)
        dataset_cfg = DatasetsConfig(**dataset_cfg)
        transform = build_transforms(dataset_cfg.test.transforms)
        dataset = build_dataset(dataset_cfg.test, transform)
        _, batch_sampler = build_sampler(dataset, phase="test", batch_size=2)
        dataloader = build_dataloader(dataset, num_workers=2, batch_sampler=batch_sampler)
        if dataset_cfg.test.batch_transforms is not None:
            batched_transform = build_batched_transform(dataset_cfg.test.batch_transforms)
        else:
            batched_transform = None
        print(f"config: {dataset_cfg}")
        print(f"dataset: {dataset}")

        for i, data in enumerate(dataloader):
            if batched_transform is not None:
                data = batched_transform(data)
            print(data)
            if i > 1:
                break


def test_lr_scheduler():
    from src.config import LrSchedulersConfig
    from src.scheduler import build_lr_scheduler

    config_dir = os.path.join(os.path.dirname(__file__), "../config/__base__/lr_scheduler")
    config_dir = Path(config_dir)

    model = torch.nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    base_cfg = _load_config(os.path.join(os.path.dirname(__file__), "../config/dummy.yaml"))
    base_cfg.epoch = 100
    for config_path in config_dir.glob("*.yaml"):
        if "None" in config_path.stem:
            continue
        scheduler_cfg = OmegaConf.load(config_path)
        scheduler_cfg = LrSchedulersConfig(**scheduler_cfg)
        base_cfg.lr_scheduler = scheduler_cfg
        print(f"config: {base_cfg.lr_scheduler}")

        iter_scheduler, scheduler = build_lr_scheduler(
            base_cfg.lr_scheduler, optimizer, base_cfg.epoch, base_cfg.max_iter
        )

        x = []
        lrs = []
        for epoch in range(base_cfg.epoch):
            for i in range(base_cfg.epoch):
                optimizer.step()
                if iter_scheduler:
                    iter_scheduler.step(
                        epoch=(i + base_cfg.epoch * epoch),
                        metric=0.1 - 0.0001 * (i + base_cfg.epoch * epoch),
                    )
                    lrs.append(optimizer.param_groups[0]["lr"])
                    x.append(epoch + i / base_cfg.epoch)
            if scheduler:
                scheduler.step(epoch=epoch, metric=0.1 - 0.0001 * epoch)
            lrs.append(optimizer.param_groups[0]["lr"])
            x.append(epoch + 1)

        scheduler_name = ""
        if scheduler_cfg.epoch_scheduler is not None:
            scheduler_name += f"{scheduler_cfg.epoch_scheduler.name}"
        if scheduler_cfg.iter_scheduler is not None:
            if scheduler_name:
                scheduler_name += "_"
            scheduler_name += f"{scheduler_cfg.iter_scheduler.name}"

        fig = plt.figure(figsize=(6, 6))
        fig.suptitle(scheduler_name, fontsize=16)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.scatter(x, lrs, s=1)
        ax1.plot(x, lrs)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("lr")
        fig.savefig(f"{os.path.dirname(__file__)}/../doc/lr_scheduler/{scheduler_name}.png")


def test_optimizer():
    from src.config import OptimizerConfig, OptimizerGroupConfig
    from src.optimizer import build_optimizer

    cfg = _load_config(os.path.join(os.path.dirname(__file__), "../config/dummy.yaml"))
    config_dir = os.path.join(os.path.dirname(__file__), "../config/__base__/optimizer")
    config_dir = Path(config_dir)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.fc(x)

    model = Model()

    for config_path in config_dir.glob("*.yaml"):
        optimizer_cfg = OmegaConf.load(config_path)
        optimizer_cfg = OptimizerConfig(**optimizer_cfg)
        cfg.optimizer = optimizer_cfg
        print(f"config: {optimizer_cfg}")

        optimizer = build_optimizer(cfg.optimizer, model)
        print(f"optimizer: {optimizer}")

        group = OptimizerGroupConfig(name="fc", divide=10)
        cfg.optimizer.group = [group]
        optimizer = build_optimizer(cfg.optimizer, model)
        print(f"grouped optimizer: {optimizer}")


def test_scripts():
    common_options = [
        os.path.join(os.path.dirname(__file__), "../config/dummy.yaml"),
        "use_cpu=true",
        "use_ram_cache=false",
    ]
    scripts = [
        "script/show_config.py",
        "script/show_model.py",
    ]

    for script in scripts:
        print("start", script)
        process = subprocess.run(
            ["python", script] + common_options,
            stdout=subprocess.PIPE,
        )
        assert process.returncode == 0
        print("end", script)


def test_backbone():
    from src.models import build_backbone
    from src.models.backbone import BackboneConfig

    # resnet18
    cfg = BackboneConfig(name="resnet18", args={"out_indices": [1, 2, 3]})
    cfg = OmegaConf.structured(cfg)
    print(cfg)
    backbone, channels = build_backbone(cfg)
    print(backbone)
    print(channels)

    # resnet18 from torchvision
    cfg = BackboneConfig(name="torchvision_resnet18", args={"out_indices": [1, 2, 3]})
    cfg = OmegaConf.structured(cfg)
    print(cfg)
    backbone, channels = build_backbone(cfg)
    print(backbone)
    print(channels)

    # swin transformer
    cfg = BackboneConfig(
        name="swin_tiny_patch4_window7_224_22k",
        pretrained=False,
        args={"out_indices": [0]},
    )
    cfg = OmegaConf.structured(cfg)
    print(cfg)
    backbone, channels = build_backbone(cfg)
    print(backbone)
    print(channels)
