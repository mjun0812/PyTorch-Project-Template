# Config API Reference

Configuration management module for hierarchical configuration handling.

## ConfigManager

::: src.config.manager.ConfigManager
    options:
      show_source: true
      show_root_heading: true

## Configuration Classes

::: src.config.config
    options:
      show_source: true
      show_root_heading: true

## Usage Examples

### Basic Configuration Loading

```python
from src.config.manager import ConfigManager

# Load configuration from file
config = ConfigManager.build("config/dummy.yaml")

# Access configuration values
print(config.dataset)  # Dataset name
print(config.model)    # Model name
print(config.optimizer.lr)  # Learning rate
```

### CLI Overrides

```python
# Simulate CLI arguments
import sys
sys.argv = ["train.py", "config/dummy.yaml", "batch=64", "optimizer.lr=0.01"]

config = ConfigManager.build()
print(config.batch)  # 64
print(config.optimizer.lr)  # 0.01
```

### Configuration Validation

```python
from src.config.config import Config

# Validate configuration structure
config_dict = {
    "dataset": "dummy_dataset",
    "model": "dummy",
    "batch": 32,
    "epoch": 100,
    "optimizer": {
        "name": "AdamW",
        "lr": 0.001
    }
}

config = Config(**config_dict)
```