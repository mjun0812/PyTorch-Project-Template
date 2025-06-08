# Models API Reference

Model implementations and builders for the PyTorch project template.

## Model Builder

::: src.models.build
    options:
      show_source: true
      show_root_heading: true

## Base Model Classes

::: src.models.base
    options:
      show_source: true
      show_root_heading: true

## Backbone Networks

### Backbone Builder

::: src.models.backbone.build
    options:
      show_source: true
      show_root_heading: true

### HuggingFace Models

::: src.models.backbone.hf_model
    options:
      show_source: true
      show_root_heading: true

### InternImage

::: src.models.backbone.internimage
    options:
      show_source: true
      show_root_heading: true

### Swin Transformer

::: src.models.backbone.swin_transformer
    options:
      show_source: true
      show_root_heading: true

## Model Layers

### Activation Functions

::: src.models.layers.activation
    options:
      show_source: true
      show_root_heading: true

### Attention Mechanisms

::: src.models.layers.attention
    options:
      show_source: true
      show_root_heading: true

### Convolution Layers

::: src.models.layers.conv
    options:
      show_source: true
      show_root_heading: true

### Normalization Layers

::: src.models.layers.norm
    options:
      show_source: true
      show_root_heading: true

### Pooling Layers

::: src.models.layers.pooling
    options:
      show_source: true
      show_root_heading: true

## Loss Functions

::: src.models.losses.build
    options:
      show_source: true
      show_root_heading: true

::: src.models.losses.base
    options:
      show_source: true
      show_root_heading: true

## Usage Examples

### Building a Model

```python
from src.models.build import build_model
from omegaconf import DictConfig

# Configuration for a model
model_config = DictConfig({
    "name": "dummy",
    "num_classes": 10,
    "pretrained": False
})

model = build_model(model_config)
```

### Custom Model Registration

```python
from src.models import MODEL_REGISTRY
import torch.nn as nn

@MODEL_REGISTRY.register("my_custom_model")
class MyCustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = build_backbone(config.backbone)
        self.classifier = nn.Linear(
            config.backbone.out_features, 
            config.num_classes
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
```

### Building Backbone Networks

```python
from src.models.backbone.build import build_backbone

backbone_config = DictConfig({
    "name": "resnet50",
    "pretrained": True,
    "num_classes": 1000
})

backbone = build_backbone(backbone_config)
```
