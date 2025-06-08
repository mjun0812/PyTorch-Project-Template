# Registry System

## Overview

The registry system provides a decoupled way to register and instantiate components dynamically. Each component type has its own registry for organization and type safety.

## Available Registries

- `MODEL_REGISTRY`: Model implementations
- `BACKBONE_REGISTRY`: Backbone networks
- `DATASET_REGISTRY`: Dataset implementations
- `OPTIMIZER_REGISTRY`: Optimizer implementations
- `SCHEDULER_REGISTRY`: Learning rate schedulers
- `TRANSFORM_REGISTRY`: Data transformations
- `EVALUATOR_REGISTRY`: Evaluation metrics

## Registration Pattern

### Basic Registration

```python
from src.utils.registry import Registry

MODEL_REGISTRY = Registry("model")

@MODEL_REGISTRY.register()
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Implementation
```

### Named Registration

```python
@MODEL_REGISTRY.register("custom_model")
class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Implementation
```

## Building Components

Components are built using the registry system:

```python
def build_model(config):
    model_cls = MODEL_REGISTRY.get(config.name)
    return model_cls(config)
```

## Configuration Integration

Registry names correspond to configuration keys:

```yaml
# Configuration
model:
  name: resnet50  # Registry key
  num_classes: 1000
  pretrained: true

# Usage
model = build_model(config.model)
```

## Registry Benefits

### Decoupling
- Components don't need to import each other
- New components can be added without modifying existing code
- Clear separation between registration and usage

### Discoverability
- All available components can be listed programmatically
- Automatic validation of component names

### Extensibility
- Easy to add new component types
- Plugin-like architecture for adding custom implementations

## Implementation Details

The registry implementation provides:

```python
class Registry:
    def register(self, name: Optional[str] = None):
        """Register a component with optional name"""
        
    def get(self, name: str):
        """Get a registered component by name"""
        
    def list(self) -> List[str]:
        """List all registered component names"""
        
    def contains(self, name: str) -> bool:
        """Check if a component is registered"""
```

## Usage Examples

### Viewing Registered Components

```bash
# Show all registered components
python script/show_registers.py

# Show specific registry
python script/show_registers.py --registry model
```

### Custom Component Registration

```python
# Custom model
@MODEL_REGISTRY.register("my_custom_model")
class MyCustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = build_backbone(config.backbone)
        self.head = nn.Linear(config.backbone.out_features, config.num_classes)

# Configuration
model:
  name: my_custom_model
  backbone:
    name: resnet50
    pretrained: true
  num_classes: 10
```