# Scheduler API Reference

Learning rate scheduler implementations and builders.

## Scheduler Builder

::: src.scheduler.build
    options:
      show_source: true
      show_root_heading: true

## Custom Schedulers

::: src.scheduler.scheduler
    options:
      show_source: true
      show_root_heading: true

## TIMM Schedulers

::: src.scheduler.timm_scheduler
    options:
      show_source: true
      show_root_heading: true

## Usage Examples

### Building a Scheduler

```python
from src.scheduler.build import build_scheduler
from src.optimizer.build import build_optimizer
from omegaconf import DictConfig
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = build_optimizer(model, {"name": "AdamW", "lr": 0.001})

# Configuration for scheduler
scheduler_config = DictConfig({
    "name": "CosineAnnealingWarmRestarts",
    "T_0": 10,
    "T_mult": 2,
    "eta_min": 1e-6
})

scheduler = build_scheduler(optimizer, scheduler_config)
```

### Available Schedulers

#### PyTorch Native Schedulers

```python
# Step LR
step_config = DictConfig({
    "name": "StepLR",
    "step_size": 30,
    "gamma": 0.1
})

# Multi-Step LR
multistep_config = DictConfig({
    "name": "MultiStepLR",
    "milestones": [30, 60, 90],
    "gamma": 0.1
})

# Cosine Annealing
cosine_config = DictConfig({
    "name": "CosineAnnealingLR",
    "T_max": 100,
    "eta_min": 1e-6
})
```

#### Custom Schedulers

```python
# Polynomial LR Decay
poly_config = DictConfig({
    "name": "PolynomialLRDecay",
    "max_decay_steps": 1000,
    "end_learning_rate": 1e-6,
    "power": 0.9
})

# Cosine Annealing with Warmup and Restarts
cosine_warmup_config = DictConfig({
    "name": "CosineAnnealingWarmupReduceRestarts",
    "T_0": 10,
    "T_mult": 2,
    "eta_min": 1e-6,
    "T_up": 2,
    "gamma": 0.5
})
```

#### TIMM Schedulers

```python
# TIMM Cosine LR
timm_cosine_config = DictConfig({
    "name": "timm_CosineLRScheduler",
    "t_initial": 100,
    "lr_min": 1e-6,
    "cycle_limit": 1,
    "t_in_epochs": True
})

# TIMM Polynomial LR
timm_poly_config = DictConfig({
    "name": "timm_PolyLRScheduler",
    "t_initial": 100,
    "lr_min": 1e-6,
    "power": 0.9,
    "t_in_epochs": True
})
```

### Chained Schedulers

```python
# Combine multiple schedulers
chained_config = DictConfig({
    "name": "ChainedScheduler",
    "schedulers": [
        {
            "name": "LinearLR", 
            "start_factor": 0.1,
            "total_iters": 10
        },
        {
            "name": "CosineAnnealingLR",
            "T_max": 90,
            "eta_min": 1e-6
        }
    ]
})
```

### Scheduler Usage in Training

```python
# Training loop with scheduler
for epoch in range(num_epochs):
    train_one_epoch(model, dataloader, optimizer)
    
    # Step the scheduler
    if hasattr(scheduler, 'step'):
        scheduler.step()
    
    # Log learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}, LR: {current_lr}")
```

### ReduceLROnPlateau

```python
# Scheduler that reduces LR based on validation metric
plateau_config = DictConfig({
    "name": "ReduceLROnPlateau",
    "mode": "min",
    "factor": 0.5,
    "patience": 10,
    "threshold": 1e-4
})

# Usage with validation loss
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    # Step with validation metric
    scheduler.step(val_loss)
```
