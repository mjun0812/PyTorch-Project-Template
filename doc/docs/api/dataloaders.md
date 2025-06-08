# Dataloaders API Reference

DataLoader and dataset implementations for efficient data handling.

## DataLoader Builder

::: src.dataloaders.build
    options:
      show_source: true
      show_root_heading: true

## Base Classes

::: src.dataloaders.base
    options:
      show_source: true
      show_root_heading: true

## Iteratable DataLoader

::: src.dataloaders.iteratable_dataloader
    options:
      show_source: true
      show_root_heading: true

## Samplers

::: src.dataloaders.sampler
    options:
      show_source: true
      show_root_heading: true

## Tensor Caching

::: src.dataloaders.tensor_cache
    options:
      show_source: true
      show_root_heading: true

## Type Definitions

::: src.dataloaders.types
    options:
      show_source: true
      show_root_heading: true

## Usage Examples

### Building a DataLoader

```python
from src.dataloaders.build import build_dataloaders
from omegaconf import DictConfig

# Configuration for dataloader
dataloader_config = DictConfig({
    "name": "dummy_dataset",
    "batch_size": 32,
    "num_workers": 4,
    "shuffle": True
})

train_loader, val_loader = build_dataloaders(dataloader_config)
```

### Custom Dataset Registration

```python
from src.dataloaders import DATASET_REGISTRY
from torch.utils.data import Dataset

@DATASET_REGISTRY.register("my_custom_dataset")
class MyCustomDataset(Dataset):
    def __init__(self, config):
        self.config = config
        # Initialize dataset
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

### Using Tensor Caching

```python
from src.dataloaders.tensor_cache import TensorCache

# Create cache for faster data loading
cache = TensorCache(
    cache_dir="./cache",
    max_memory_gb=8.0
)

# Cache tensors
tensor_data = torch.randn(1000, 3, 224, 224)
cache.save("train_images", tensor_data)

# Load cached tensors
loaded_data = cache.load("train_images")
```

### Iteratable DataLoader for Large Datasets

```python
from src.dataloaders.iteratable_dataloader import IteratableDataLoader

# For streaming large datasets
dataloader = IteratableDataLoader(
    dataset=streaming_dataset,
    batch_size=32,
    num_workers=4
)

for batch in dataloader:
    # Process batch
    pass
```
