# Custom Components

Learn how to extend the PyTorch Project Template with your own models, datasets, optimizers, and other components.

## Component Registration System

All components use the registry pattern for registration and discovery:

```python
from src.utils.registry import Registry

# Each component type has its own registry
MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")
OPTIMIZER_REGISTRY = Registry("optimizer")
# ... and more
```

## Creating Custom Models

### Basic Model Registration

```python
# src/models/my_models.py
from src.models import MODEL_REGISTRY
import torch.nn as nn

@MODEL_REGISTRY.register()
class MyCustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.hidden_dim = config.get("hidden_dim", 512)
        
        self.backbone = nn.Sequential(
            nn.Linear(config.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
```

### Configuration for Custom Model

```yaml
# config/my_experiment.yaml
model:
  name: MyCustomModel
  input_dim: 784
  hidden_dim: 512
  num_classes: 10
```

### Advanced Model with Backbone

```python
# src/models/my_advanced_model.py
from src.models import MODEL_REGISTRY
from src.models.backbone.build import build_backbone
import torch.nn as nn

@MODEL_REGISTRY.register()
class MyAdvancedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = build_backbone(config.backbone)
        self.num_classes = config.num_classes
        
        # Get backbone output dimension
        backbone_dim = self.backbone.num_features
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(backbone_dim, self.num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
```

```yaml
# Configuration with backbone
model:
  name: MyAdvancedModel
  num_classes: 1000
  dropout: 0.2
  backbone:
    name: resnet50
    pretrained: true
```

## Creating Custom Datasets

### Basic Dataset

```python
# src/dataloaders/my_dataset.py
from src.dataloaders import DATASET_REGISTRY
from torch.utils.data import Dataset
import torch
import numpy as np

@DATASET_REGISTRY.register()
class MyCustomDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_path = config.data_path
        self.split = config.get("split", "train")
        
        # Load your data
        self.data, self.labels = self.load_data()
        
        # Apply transforms if provided
        self.transform = config.get("transform", None)
    
    def load_data(self):
        # Implement your data loading logic
        if self.split == "train":
            data = np.random.randn(1000, 3, 224, 224)
            labels = np.random.randint(0, 10, 1000)
        else:
            data = np.random.randn(200, 3, 224, 224)
            labels = np.random.randint(0, 10, 200)
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

### Advanced Dataset with Caching

```python
# src/dataloaders/cached_dataset.py
from src.dataloaders import DATASET_REGISTRY
from src.dataloaders.tensor_cache import TensorCache
from torch.utils.data import Dataset
import os

@DATASET_REGISTRY.register()
class CachedDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.cache_dir = config.get("cache_dir", "./cache")
        
        # Initialize cache
        self.cache = TensorCache(
            cache_dir=self.cache_dir,
            max_memory_gb=config.get("max_cache_memory", 4.0)
        )
        
        # Load or create cached data
        self.setup_cache()
    
    def setup_cache(self):
        cache_key = f"{self.config.split}_{self.config.version}"
        
        if self.cache.exists(cache_key):
            self.data, self.labels = self.cache.load(cache_key)
        else:
            # Load and preprocess data
            self.data, self.labels = self.load_and_preprocess()
            # Cache for future use
            self.cache.save(cache_key, (self.data, self.labels))
    
    def load_and_preprocess(self):
        # Expensive data loading and preprocessing
        pass
```

## Creating Custom Optimizers

### Basic Optimizer

```python
# src/optimizer/my_optimizer.py
from src.optimizer import OPTIMIZER_REGISTRY
import torch.optim as optim

@OPTIMIZER_REGISTRY.register()
class MyCustomOptimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Custom optimization step
                p.data.add_(grad, alpha=-group['lr'])
        
        return loss
```

### Optimizer Builder Function

```python
# src/optimizer/my_optimizer.py (continued)
@OPTIMIZER_REGISTRY.register()
def build_my_optimizer(model, config):
    """Build optimizer with custom parameter groups."""
    
    # Separate parameters by type
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    # Different learning rates for different parts
    param_groups = [
        {"params": backbone_params, "lr": config.lr * 0.1},
        {"params": head_params, "lr": config.lr}
    ]
    
    return MyCustomOptimizer(param_groups, **config)
```

## Creating Custom Transforms

### Basic Transform

```python
# src/transform/my_transforms.py
from src.transform import TRANSFORM_REGISTRY
import torch
import numpy as np

@TRANSFORM_REGISTRY.register()
class MyCustomTransform:
    def __init__(self, config):
        self.strength = config.get("strength", 1.0)
        self.probability = config.get("probability", 0.5)
    
    def __call__(self, data):
        if np.random.random() < self.probability:
            # Apply your custom transformation
            return self.apply_transform(data)
        return data
    
    def apply_transform(self, data):
        # Implement your transformation logic
        noise = torch.randn_like(data) * self.strength * 0.1
        return data + noise
```

### Composition Transform

```python
@TRANSFORM_REGISTRY.register()
class MyCompositeTransform:
    def __init__(self, config):
        self.transforms = []
        for transform_config in config.transforms:
            transform = build_transform(transform_config)
            self.transforms.append(transform)
    
    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data
```

## Creating Custom Evaluators

### Basic Evaluator

```python
# src/evaluator/my_evaluator.py
from src.evaluator import EVALUATOR_REGISTRY
from src.evaluator.base import BaseEvaluator
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

@EVALUATOR_REGISTRY.register()
class MyCustomEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config.num_classes
        self.metrics = config.get("metrics", ["accuracy", "f1_score"])
    
    def evaluate(self, predictions, targets):
        """
        Evaluate predictions against targets.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth targets
            
        Returns:
            Dictionary of metric results
        """
        # Convert to numpy for sklearn compatibility
        if torch.is_tensor(predictions):
            pred_labels = predictions.argmax(dim=1).cpu().numpy()
        else:
            pred_labels = np.argmax(predictions, axis=1)
        
        if torch.is_tensor(targets):
            true_labels = targets.cpu().numpy()
        else:
            true_labels = targets
        
        results = {}
        
        if "accuracy" in self.metrics:
            results["accuracy"] = accuracy_score(true_labels, pred_labels)
        
        if "f1_score" in self.metrics:
            results["f1_score"] = f1_score(
                true_labels, pred_labels, 
                average="macro", 
                zero_division=0
            )
        
        # Custom metrics
        if "per_class_accuracy" in self.metrics:
            results.update(self.compute_per_class_accuracy(true_labels, pred_labels))
        
        return results
    
    def compute_per_class_accuracy(self, true_labels, pred_labels):
        """Compute per-class accuracy."""
        per_class_acc = {}
        for class_id in range(self.num_classes):
            mask = true_labels == class_id
            if mask.sum() > 0:
                class_acc = (pred_labels[mask] == true_labels[mask]).mean()
                per_class_acc[f"accuracy_class_{class_id}"] = class_acc
        return per_class_acc
```

## Creating Custom Schedulers

### Basic Scheduler

```python
# src/scheduler/my_scheduler.py
from src.scheduler import SCHEDULER_REGISTRY
import torch.optim as optim
import math

@SCHEDULER_REGISTRY.register()
class MyCustomScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps=1000, total_steps=10000, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch / self.warmup_steps) 
                    for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [base_lr * 0.5 * (1 + math.cos(math.pi * progress)) 
                    for base_lr in self.base_lrs]
```

## Integration and Configuration

### Registering Components

Make sure to import your custom components so they get registered:

```python
# src/__init__.py or src/models/__init__.py
from .my_models import MyCustomModel, MyAdvancedModel
from .my_dataset import MyCustomDataset, CachedDataset
from .my_optimizer import MyCustomOptimizer
from .my_transforms import MyCustomTransform
from .my_evaluator import MyCustomEvaluator
from .my_scheduler import MyCustomScheduler
```

### Configuration Examples

```yaml
# config/my_custom_experiment.yaml
dataset:
  name: MyCustomDataset
  data_path: "/path/to/data"
  cache_dir: "./cache"
  max_cache_memory: 8.0

model:
  name: MyAdvancedModel
  num_classes: 100
  dropout: 0.2
  backbone:
    name: resnet50
    pretrained: true

optimizer:
  name: MyCustomOptimizer
  lr: 0.001
  weight_decay: 0.01

lr_scheduler:
  name: MyCustomScheduler
  warmup_steps: 1000
  total_steps: 10000

evaluator:
  name: MyCustomEvaluator
  num_classes: 100
  metrics: ["accuracy", "f1_score", "per_class_accuracy"]

transform:
  name: Compose
  transforms:
    - name: MyCustomTransform
      strength: 1.0
      probability: 0.5
    - name: ToTensor
    - name: Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

## Best Practices

1. **Follow Naming Conventions**: Use descriptive names for your components
2. **Configuration Validation**: Validate configuration parameters in `__init__`
3. **Documentation**: Add docstrings to all custom components
4. **Testing**: Write unit tests for your custom components
5. **Error Handling**: Implement proper error handling and validation
6. **Performance**: Consider caching and optimization for expensive operations
7. **Compatibility**: Ensure compatibility with distributed training and mixed precision

## Testing Custom Components

```python
# tests/test_custom_components.py
import pytest
from omegaconf import DictConfig
from src.models.build import build_model
from src.dataloaders.build import build_dataset

def test_custom_model():
    config = DictConfig({
        "name": "MyCustomModel",
        "input_dim": 784,
        "hidden_dim": 512,
        "num_classes": 10
    })
    
    model = build_model(config)
    assert model is not None
    
    # Test forward pass
    import torch
    x = torch.randn(4, 784)
    output = model(x)
    assert output.shape == (4, 10)

def test_custom_dataset():
    config = DictConfig({
        "name": "MyCustomDataset",
        "data_path": "/tmp/test_data",
        "split": "train"
    })
    
    dataset = build_dataset(config)
    assert len(dataset) > 0
    
    # Test data loading
    sample = dataset[0]
    assert len(sample) == 2  # (data, label)
```

This guide provides a comprehensive overview of creating custom components. Each component type follows similar patterns, making it easy to extend the framework with your own implementations.