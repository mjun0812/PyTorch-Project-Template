# Configuration System

## Overview

The configuration system uses OmegaConf and dataclasses to provide hierarchical, type-safe configuration management with CLI override support.

## Configuration Structure

```yaml
# Base configuration structure
dataset: dummy_dataset
model: dummy
batch: 32
epoch: 100

optimizer:
  name: AdamW
  lr: 0.001
  weight_decay: 0.01

lr_scheduler:
  name: CosineAnnealingWarmRestarts
  T_0: 10
  T_mult: 2

gpu:
  use: "0"
  mixed_precision: true
```

## Configuration Hierarchy

### Base Configuration
Located in `config/__base__/config.yaml`, contains default values for all components.

### Component Configurations
Individual YAML files for each component type:
- `config/__base__/optimizer/`: Optimizer configurations
- `config/__base__/lr_scheduler/`: Scheduler configurations
- `config/__base__/dataset/`: Dataset configurations

### Environment-Specific Configurations
Project-specific configurations in `config/` directory override base settings.

## CLI Overrides

Configuration values can be overridden via command line:

```bash
# Override batch size and learning rate
python train.py config/dummy.yaml batch=64 optimizer.lr=0.01

# Override GPU settings
python train.py config/dummy.yaml gpu.use="0,1,2,3" gpu.mixed_precision=false
```

## Configuration Loading Process

1. **Base Loading**: Load base configuration from `__base__/config.yaml`
2. **Component Merging**: Merge component-specific configurations
3. **Override Application**: Apply project-specific and CLI overrides
4. **Validation**: Validate configuration against dataclass schemas

## Configuration Dataclasses

Type-safe configuration using dataclasses:

```python
@dataclass
class OptimizerConfig:
    name: str
    lr: float = 0.001
    weight_decay: float = 0.01

@dataclass
class Config:
    dataset: str
    model: str
    batch: int
    epoch: int
    optimizer: OptimizerConfig
```

## Environment Variables

Configuration supports environment variable substitution:

```yaml
# Using environment variables
model:
  name: ${MODEL_NAME:resnet50}
  pretrained: ${PRETRAINED:true}
```