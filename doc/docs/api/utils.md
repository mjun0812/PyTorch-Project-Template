# Utils API Reference

Utility functions and helper classes for the PyTorch project template.

## Registry System

::: src.utils.registry
    options:
      show_source: true
      show_root_heading: true

## Logger

::: src.utils.logger
    options:
      show_source: true
      show_root_heading: true

## PyTorch Utilities

::: src.utils.torch_utils
    options:
      show_source: true
      show_root_heading: true

## General Utilities

::: src.utils.utils
    options:
      show_source: true
      show_root_heading: true

## Usage Examples

### Registry System

```python
from src.utils.registry import Registry

# Create a new registry
MY_REGISTRY = Registry("my_components")

# Register a component
@MY_REGISTRY.register()
class MyComponent:
    def __init__(self, config):
        self.config = config

# Register with custom name
@MY_REGISTRY.register("custom_name")
class AnotherComponent:
    def __init__(self, config):
        self.config = config

# Get registered component
component_cls = MY_REGISTRY.get("MyComponent")
component = component_cls(config)

# List all registered components
available = MY_REGISTRY.list()
print(f"Available components: {available}")
```

### Logger Configuration

```python
from src.utils.logger import setup_logger
import logging

# Setup logger with custom configuration
logger = setup_logger(
    name="training",
    level=logging.INFO,
    log_file="training.log"
)

# Use logger
logger.info("Starting training...")
logger.warning("Learning rate might be too high")
logger.error("Training failed due to CUDA error")
```

### PyTorch Utilities

```python
from src.utils.torch_utils import (
    count_parameters,
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint
)
import torch.nn as nn

# Count model parameters
model = nn.Linear(100, 10)
total_params = count_parameters(model)
print(f"Model has {total_params} parameters")

# Set random seed for reproducibility
set_seed(42)

# Get appropriate device
device = get_device()
model = model.to(device)

# Save model checkpoint
save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=10,
    loss=0.1,
    filepath="checkpoint.pth"
)

# Load model checkpoint
checkpoint = load_checkpoint("checkpoint.pth")
model.load_state_dict(checkpoint["model_state_dict"])
```

### General Utilities

```python
from src.utils.utils import (
    ensure_dir,
    get_timestamp,
    flatten_dict,
    unflatten_dict
)
import os

# Ensure directory exists
ensure_dir("results/experiment_1")

# Get current timestamp
timestamp = get_timestamp()
print(f"Experiment started at: {timestamp}")

# Flatten nested dictionary
nested_dict = {
    "model": {"name": "resnet50", "layers": 50},
    "optimizer": {"lr": 0.001, "weight_decay": 0.01}
}
flat_dict = flatten_dict(nested_dict)
# Result: {"model.name": "resnet50", "model.layers": 50, ...}

# Unflatten dictionary
original_dict = unflatten_dict(flat_dict)
```

### Configuration Utilities

```python
from omegaconf import DictConfig, OmegaConf

# Save configuration
config = DictConfig({
    "model": {"name": "resnet50"},
    "training": {"epochs": 100, "lr": 0.001}
})

# Save to file
OmegaConf.save(config, "experiment_config.yaml")

# Load configuration
loaded_config = OmegaConf.load("experiment_config.yaml")
```

### Memory and Performance Utilities

```python
from src.utils.torch_utils import get_memory_usage
import torch

# Monitor GPU memory usage
if torch.cuda.is_available():
    memory_info = get_memory_usage()
    print(f"GPU Memory: {memory_info['used']:.1f}GB / {memory_info['total']:.1f}GB")

# Profile model inference
from src.utils.torch_utils import profile_model

input_tensor = torch.randn(1, 3, 224, 224)
flops, params = profile_model(model, input_tensor)
print(f"Model FLOPs: {flops}, Parameters: {params}")
```

### Distributed Training Utilities

```python
from src.utils.torch_utils import (
    init_distributed,
    cleanup_distributed,
    is_distributed,
    get_rank,
    get_world_size
)

# Initialize distributed training
if is_distributed():
    init_distributed()
    
    rank = get_rank()
    world_size = get_world_size()
    print(f"Process {rank}/{world_size}")

# Cleanup distributed training
cleanup_distributed()
```

### Data Utilities

```python
from src.utils.utils import (
    split_dataset,
    compute_dataset_stats,
    normalize_data
)

# Split dataset into train/val/test
train_data, val_data, test_data = split_dataset(
    dataset, 
    splits=[0.7, 0.2, 0.1],
    random_state=42
)

# Compute dataset statistics
stats = compute_dataset_stats(train_data)
print(f"Mean: {stats['mean']}, Std: {stats['std']}")

# Normalize data using computed statistics
normalized_data = normalize_data(data, stats['mean'], stats['std'])
```