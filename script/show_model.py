import torch

from src.config import ConfigManager, DatasetConfig, ExperimentConfig
from src.dataloaders import build_dataloader, build_dataset, build_sampler
from src.models import build_model
from src.transform import (
    build_batched_transform,
    build_transforms,
)


@ConfigManager.argparse
def main(cfg: ExperimentConfig):
    cfg.model.checkpoint = None

    if cfg.use_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    model = build_model(cfg.model, phase="train")
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    print(model)

    for phase in ["train", "val", "test"]:
        print(f"Phase: {phase}")
        model.phase = phase

        cfg_dataset: DatasetConfig = cfg.dataset.get(phase)

        transform = build_transforms(cfg_dataset.transforms)
        if cfg_dataset.batch_transforms is not None:
            batched_transform = build_batched_transform(cfg_dataset.batch_transforms)
        else:
            batched_transform = None

        dataset = build_dataset(cfg_dataset, transform)
        _, batch_sampler = build_sampler(dataset, phase, cfg.batch, cfg.dataset.batch_sampler)
        dataloader = build_dataloader(dataset, cfg.num_worker, batch_sampler)

        data = next(iter(dataloader))
        if batched_transform is not None:
            data = batched_transform(data)
        print(f"Data: {data}")
        y = model(data)
        print(f"Output: {y}")


if __name__ == "__main__":
    main()
