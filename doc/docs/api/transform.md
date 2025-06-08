# Transform API Reference

Data transformation and augmentation pipeline for preprocessing.

## Transform Builder

::: src.transform.build
    options:
      show_source: true
      show_root_heading: true

## Base Classes

::: src.transform.base
    options:
      show_source: true
      show_root_heading: true

## Transform Composition

::: src.transform.compose
    options:
      show_source: true
      show_root_heading: true

## Data Conversion

::: src.transform.convert
    options:
      show_source: true
      show_root_heading: true

## Multiple Transforms

::: src.transform.multiple
    options:
      show_source: true
      show_root_heading: true

## Usage Examples

### Building Transforms

```python
from src.transform.build import build_transform
from omegaconf import DictConfig

# Configuration for transforms
transform_config = DictConfig({
    "name": "Compose",
    "transforms": [
        {"name": "Resize", "size": [224, 224]},
        {"name": "RandomHorizontalFlip", "p": 0.5},
        {"name": "ToTensor"},
        {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    ]
})

transform = build_transform(transform_config)
```

### Custom Transform Registration

```python
from src.transform import TRANSFORM_REGISTRY
import torch
import numpy as np

@TRANSFORM_REGISTRY.register("my_custom_transform")
class MyCustomTransform:
    def __init__(self, config):
        self.param = config.param
        
    def __call__(self, data):
        # Apply custom transformation
        return transformed_data
```

### Common Transform Pipelines

#### Training Pipeline

```python
train_transform_config = DictConfig({
    "name": "Compose",
    "transforms": [
        {"name": "Resize", "size": [256, 256]},
        {"name": "RandomCrop", "size": [224, 224]},
        {"name": "RandomHorizontalFlip", "p": 0.5},
        {"name": "RandomRotation", "degrees": 10},
        {"name": "ColorJitter", "brightness": 0.2, "contrast": 0.2},
        {"name": "ToTensor"},
        {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    ]
})
```

#### Validation Pipeline

```python
val_transform_config = DictConfig({
    "name": "Compose",
    "transforms": [
        {"name": "Resize", "size": [256, 256]},
        {"name": "CenterCrop", "size": [224, 224]},
        {"name": "ToTensor"},
        {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    ]
})
```

### Multiple Transform Outputs

```python
# Apply multiple transforms to the same input
multi_transform_config = DictConfig({
    "name": "MultipleTransforms",
    "transforms": [
        {"name": "WeakAugmentation", "transforms": [...]},
        {"name": "StrongAugmentation", "transforms": [...]}
    ]
})

# Returns multiple transformed versions
weak_aug, strong_aug = multi_transform(image)
```

### Data Type Conversion

```python
# Convert between different data formats
convert_config = DictConfig({
    "name": "Convert",
    "input_format": "PIL",
    "output_format": "tensor",
    "dtype": "float32"
})

converter = build_transform(convert_config)
tensor_data = converter(pil_image)
```

### Transform Composition Strategies

#### Sequential Composition

```python
compose_config = DictConfig({
    "name": "Compose",
    "transforms": [
        {"name": "Transform1", ...},
        {"name": "Transform2", ...},
        {"name": "Transform3", ...}
    ]
})
```

#### Conditional Transforms

```python
conditional_config = DictConfig({
    "name": "RandomChoice",
    "transforms": [
        {"name": "Transform1", "weight": 0.3},
        {"name": "Transform2", "weight": 0.7}
    ]
})
```

### Integration with DataLoaders

```python
from src.dataloaders.build import build_dataloaders

# Combine with dataset configuration
dataset_config = DictConfig({
    "name": "ImageDataset",
    "data_path": "/path/to/data",
    "transform": {
        "name": "Compose",
        "transforms": [...]
    }
})

train_loader, val_loader = build_dataloaders(dataset_config)
```