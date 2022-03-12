# PyTorch Project Template

My PyTorch Project Template.

## Environments

Python >= 3.7
PyTorch >= 1.8

## Features

- Multi GPU Training
- Use [Hydra](https://github.com/facebookresearch/hydra) Config file management (YAML)
- Continue Training from your own weight
- Early Stopping
- Tensorboard Logging
- Easy additional Model, Loss, Transform(Augmentation) implementation

## Multi GPU

Change Setting in `config/config.yaml`(below).

```yaml
# Device Params
GPU:
  USE: 0
  MULTI: False
CPU: False
```

If you use multi GPU, set `GPU.MULTI: True` and `GPU.USE: "0,1,2,3"`.

This setting can change from CLI using Hydra.

```bash
python train.py GPU.MULTI=True GPU.USE="'1,2'"
```

## Required Editing

```bash
pytorch-project-template
├── config
│   └── config.yaml
├── log_parse.py
├── requirements.txt
├── src
│   ├── dataloaders
│   │   ├── __init__.py
│   │   └── dataloader.py
│   ├── losses
│   │   ├── __init__.py
│   │   ├── build.py
│   │   └── loss.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── build.py
│   │   └── model.py
│   ├── transform.py
├── test.py
└── train.py
```

### config

This Template uses [Hydra](https://github.com/facebookresearch/hydra).  
Hydra supports multi config file loading.

Example:

```bash
├── config
│   ├── config.yaml
│   └── DATASET
│        ├── coco.yaml
│        └── pascalVOC.yaml
```

```yaml
# config.yaml
defaults:
  - _self_
  - DATASET: coco
```

```yaml
# DATASET/coco.yaml
NAME: dataset_name
TRAIN_LIST: ../dataset/coco128/annotations/annotations.csv
VAL_LIST: ../dataset/coco128/annotations/annotations.csv
TEST_LIST: ../dataset/coco128/annotations/annotations.csv
CLASS_LIST: ../dataset/coco128/annotations/classes.csv
NUM_CLASSES: 80
TRANSFORMS:
  TRAIN:
    - name: RandomLRFlip
      args:
        width: 1.0
        num_point_feature: ${DATASET.NUM_POINT_FEATURE}
  VAL:
    - name: RandomLRFlip
      args:
        width: 1.0
        num_point_feature: ${DATASET.NUM_POINT_FEATURE}
  TEST:
    - name: TransformStGCN
      args:
        frame: ${DATASET.FRAME}
        point_num: 33
```

You can access yaml value from `cfg.DATASET.NUM_CLASSES` in code.  
Example of change config file from CLI is below.

```bash
python train.py DATASET=coco
```

### train.py

Change your implemantion Dataset class.

```python
def build_dataset(cfg, phase="train"):
    if phase == "train":
        filelist = cfg.DATASET.TRAIN_LIST
    elif phase == "val":
        filelist = cfg.DATASET.VAL_LIST
    elif phase == "test":
        filelist = cfg.DATASET.TEST_LIST

    transform = build_transforms(cfg, phase=phase)
    dataset = Dataset(cfg, filelist)
    logger.info(f"{phase.capitalize()} Dataset sample num: {len(dataset)}")
    logger.info(f"{phase.capitalize()} transform: {transform}")
    return dataset
```

Change your model input format and calc loss.

```python
for i, data in progress_bar:
    # Edit below
    data = data.to(device, non_blocking=True).float()

    optimizer.zero_grad()

    # Calculate Loss
    y = model(data)

    # Edit below
    loss = loss_fn(y)

    if loss == 0 or not torch.isfinite(loss):
        continue
    hist_epoch_loss += loss * data.size(0)
    if rank in [-1, 0]:
        progress_bar.set_description(f"Epoch: {epoch + 1}/{max_epoch}. Loss: {loss.item():.5f}")

    loss.backward()
    optimizer.step()

    del loss
```

```python
def do_validate(model, dataloader, loss_fn):
    hist_loss = []
    model.eval()
    device = next(model.parameters()).device
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            # Edit below
            data = data.to(device, non_blocking=True).float()

            y = model(data)

            # Edit below
            loss = loss_fn(y)

            if loss == 0 or not torch.isfinite(loss):
                continue
            hist_loss.append(loss.item())
    loss = np.mean(hist_loss)
    return loss
```

### test.py

Add metrics and change input format.

```python
def do_test(cfg, output_dir, device):
    logger.info("Loading Dataset...")
    dataset = build_dataset(cfg, phase="test")

    logger.info(f"Load model weight {cfg.MODEL.WEIGHT}")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(cfg.MODEL.WEIGHT, map_location=device))
    logger.info("Complete load model")

    inference_speed = 0
    metric = 0
    results = []
    model.requires_grad_(False)
    model.eval()
    progress_bar = tqdm(enumerate(dataset), total=len(dataset))
    for i, data in progress_bar:
        with torch.no_grad():
            # Edit below
            input_data = data.to(device).unsqueeze(dim=0)

            t = time_synchronized()
            y = model(input_data)
            inference_speed += time_synchronized() - t

            # calc metrics below
            # Edit below
            result = y
            results.append(result)

    inference_speed /= len(dataset)
    logger.info(f"Average Inferance Speed: {inference_speed:.5f}s, {(1.0 / inference_speed):.2f}fps")

    # 評価結果の保存
    with open(os.path.join(output_dir, "result.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)
    return metric
```

### models

Add model.

```python
# src/models/__init__.py
from .build import MODEL_REGISTRY, build_model

# Add import
from .model import Model

__all__ = list(globals().keys())

```

```python
# src/models/model.py
import torch.nn as nn

from .build import MODEL_REGISTRY

# Below line add top of model class.
@MODEL_REGISTRY.register()
class Model(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg

    def forward(self, x):
        return x
```

### losses

Add loss.

```python
# src/losses/__init__.py
from .build import LOSS_REGISTRY, build_loss

# Add import
from .loss import loss

__all__ = list(globals().keys())
```

```python
# src/losses/loss.py
import torch.nn.functional as F

from .build import LOSS_REGISTRY

# Below line add top of loss class or function.
@LOSS_REGISTRY.register()
def loss(x):
    return F.mse_loss(x)
```

### dataloaders

Add your own Dataset class.

```python
# src/dataloaders/dataloader.py
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return idx
```

### transforms

Add Transforms.

```python
# src/transform.py
from torchvision import transforms

from .utils.registry import Registry

TRANSFORM_REGISTRY = Registry("TRANSFORM")


def build_transforms(cfg, phase="train"):
    cfg = cfg.DATASET.TRANSFORMS
    if phase == "train":
        cfg = cfg.TRAIN
    elif phase == "val":
        cfg = cfg.VAL
    elif phase == "test":
        cfg = cfg.TEST

    transes = []
    for trans in cfg:
        if trans.args is None:
            transform = TRANSFORM_REGISTRY.get(trans.name)()
        else:
            params = {k: v for k, v in trans.args.items()}
            transform = TRANSFORM_REGISTRY.get(trans.name)(**params)
        transes.append(transform)
    return transforms.Compose(transes)


@TRANSFORM_REGISTRY.register()
def list_to_tensor(list_obj, label):
    return torch.tensor(list_obj, dtype=torch.float64), torch.tensor(
        label, dtype=torch.float64
    )
```
