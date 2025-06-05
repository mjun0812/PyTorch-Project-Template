from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torch import optim

from src.config import ConfigManager, ExperimentConfig
from src.models.backbone import BackboneConfig, build_backbone, get_available_backbones


def _load_config(path: str) -> ExperimentConfig:
    # From Dataclass
    cfg = OmegaConf.structured(ExperimentConfig)
    # From File
    cfg_from_file = ConfigManager.build_config_from_file(path)
    cfg = ConfigManager.merge(cfg, cfg_from_file)
    return cfg


def test_config() -> None:
    cfg = _load_config(Path(__file__).parent.parent / "config/dummy.yaml")
    print(ConfigManager.pretty_text(cfg))


def test_available_backbones() -> None:
    """利用可能なバックボーンモデルのリストをテスト"""
    available_backbones = get_available_backbones()
    assert len(available_backbones) > 0
    print(available_backbones[:10])


def test_torchvision_backbone() -> None:
    """torchvisionモデルの構築をテスト"""
    cfg = BackboneConfig(
        name="torchvision_resnet18",
        pretrained=False,  # テスト高速化のため事前学習なし
        args=OmegaConf.create({"out_indices": [1, 2, 3, 4]}),
    )

    backbone, _ = build_backbone(cfg)

    # モデルが入力を処理できるか確認（小さいサイズの入力を使用）
    dummy_input = torch.randn(1, 3, 224, 224)
    outputs = backbone(dummy_input)
    assert len(outputs) == 4


def test_timm_backbone() -> None:
    """timmモデルの構築をテスト(GPUなしでも実行可能)"""
    cfg = BackboneConfig(
        name="resnet18",
        pretrained=False,  # テスト高速化のため事前学習なし
        args=OmegaConf.create({"out_indices": [1, 2, 3, 4]}),
    )

    backbone, _ = build_backbone(cfg)

    # モデルが入力を処理できるか確認（小さいサイズの入力を使用）
    dummy_input = torch.randn(1, 3, 224, 224)
    outputs = backbone(dummy_input)
    assert len(outputs) == 4


def test_internimage_backbone() -> None:
    """InternImageモデルの構築をテスト"""
    cfg = BackboneConfig(
        name="internimage_t_1k_224",
        pretrained=False,  # テスト高速化のため事前学習なし
        args=OmegaConf.create({"out_indices": [0, 1, 2, 3]}),
    )

    backbone, _ = build_backbone(cfg)

    # モデルが入力を処理できるか確認（小さいサイズの入力を使用）
    dummy_input = torch.randn(1, 3, 224, 224)
    outputs = backbone(dummy_input)
    assert len(outputs) == 4


def test_backbone() -> None:
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


def test_dataloader() -> None:
    from src.config import DatasetsConfig
    from src.dataloaders import build_dataloader, build_dataset, build_sampler
    from src.transform import build_batched_transform, build_transforms

    config_dir = Path(__file__).parent.parent / "config/__base__/dataset"
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


def test_lr_scheduler() -> None:
    from src.config import LrSchedulersConfig
    from src.scheduler import build_lr_scheduler

    config_dir = Path(__file__).parent.parent / "config/__base__/lr_scheduler"

    model = torch.nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    base_cfg = _load_config(Path(__file__).parent.parent / "config/dummy.yaml")
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
        fig.savefig(Path(__file__).parent.parent / f"doc/lr_scheduler/{scheduler_name}.png")


def test_optimizer() -> None:
    from src.config import OptimizerConfig, OptimizerGroupConfig
    from src.optimizer import build_optimizer

    cfg = _load_config(Path(__file__).parent.parent / "config/dummy.yaml")
    config_dir = Path(__file__).parent.parent / "config/__base__/optimizer"

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
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
