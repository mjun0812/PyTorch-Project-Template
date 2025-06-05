import torch

from src.config import ConfigManager, ExperimentConfig
from src.dataloaders import build_dataset
from src.evaluator import build_evaluator
from src.models import build_model


@ConfigManager.argparse
def main(cfg: ExperimentConfig) -> None:
    cfg.batch = 2
    phase = "test"
    cfg.model.pre_trained_weight = None

    device = torch.device("cpu" if cfg.use_cpu else "cuda:0")

    data = build_dataset(cfg, phase)
    _, dataloader, batched_transforms = data

    model, _ = build_model(cfg, device=device, phase=phase)
    model.eval()
    model.requires_grad_(False)

    evaluator = build_evaluator(cfg, phase)
    if evaluator is not None:
        evaluator = evaluator.to(device)

    for i, data in enumerate(dataloader):
        if batched_transforms:
            data = batched_transforms(data)
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.float().to(device, non_blocking=True)
        output = model(data)
        if evaluator is not None:
            evaluator.update(data, output["preds"])
        break

    metrics = evaluator.compute()
    evaluator.reset()
    print(metrics)


if __name__ == "__main__":
    main()
