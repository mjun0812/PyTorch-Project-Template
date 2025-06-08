# Optimizer API Reference

Optimizer implementations and builders for training optimization.

## Optimizer Builder

::: src.optimizer.build
    options:
      show_source: true
      show_root_heading: true

## PyTorch Optimizers

::: src.optimizer.torch_optimizers
    options:
      show_source: true
      show_root_heading: true

## Custom Optimizers

### Lion Optimizer

::: src.optimizer.lion
    options:
      show_source: true
      show_root_heading: true

### Schedule-Free Optimizers

::: src.optimizer.schedule_free
    options:
      show_source: true
      show_root_heading: true

## TIMM Optimizers

::: src.optimizer.timm_optimizer
    options:
      show_source: true
      show_root_heading: true

## Usage Examples

### Building an Optimizer

```python
from src.optimizer.build import build_optimizer
from omegaconf import DictConfig
import torch.nn as nn

model = nn.Linear(10, 1)

# Configuration for optimizer
optimizer_config = DictConfig({
    "name": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999]
})

optimizer = build_optimizer(model, optimizer_config)
```

### Using Custom Optimizers

```python
# Lion optimizer configuration
lion_config = DictConfig({
    "name": "Lion",
    "lr": 0.0001,
    "betas": [0.9, 0.99],
    "weight_decay": 0.01
})

lion_optimizer = build_optimizer(model, lion_config)
```

### Schedule-Free Optimizers

```python
# AdamW Schedule-Free configuration
schedule_free_config = DictConfig({
    "name": "AdamWScheduleFree",
    "lr": 0.001,
    "betas": [0.9, 0.999],
    "weight_decay": 0.01,
    "warmup_steps": 1000
})

schedule_free_optimizer = build_optimizer(model, schedule_free_config)
```

### Parameter Groups

```python
# Different learning rates for different layers
backbone_params = model.backbone.parameters()
head_params = model.head.parameters()

optimizer_config = DictConfig({
    "name": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.01
})

# Custom parameter groups
param_groups = [
    {"params": backbone_params, "lr": 0.0001},
    {"params": head_params, "lr": 0.001}
]

optimizer = build_optimizer(param_groups, optimizer_config)
```

### Available Optimizers

The following optimizers are available through the registry:

- **PyTorch Native**: Adam, AdamW, SGD, RMSprop, etc.
- **Custom Implementations**: Lion, Schedule-Free variants
- **TIMM Optimizers**: Various optimizers from the timm library

Each optimizer supports standard PyTorch optimizer interface with additional configuration options specific to the implementation.
