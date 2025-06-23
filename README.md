# PyTorch Project Template

A comprehensive, production-ready PyTorch project template with modular architecture, distributed training support, and modern tooling.

## Features

- **🧩 Modular Architecture**: Registry-based component system for easy extensibility
- **⚙️ Configuration Management**: Hierarchical config system with inheritance and CLI overrides
- **🚀 Distributed Training**: Multi-node/multi-GPU training with DDP, FSDP, and DataParallel
- **📊 Experiment Tracking**: MLflow and Weights & Biases integration with auto-visualization
- **🔧 Modern Tooling**: uv package management, pre-commit hooks, Docker support
- **💾 Resume Training**: Automatic checkpoint saving and loading with state preservation
- **🌐 Cross-Platform**: Development support on macOS and Linux with optimized builds
- **🐳 Development Environment**: Devcontainer and Jupyter Lab integration
- **⚡ Performance Optimization**: RAM caching, mixed precision, torch.compile support
- **📚 Auto Documentation**: Sphinx-based API docs with live reloading
- **📱 Slack Notifications**: Training completion and error notifications
- **🛡️ Error Handling**: Robust error recovery and automatic retries

## Requirements

- **Python**: 3.11+
- **Package Manager**: uv
- **CUDA**: 12.8 (for GPU training)
- **PyTorch**: 2.7.1

## Quick Start

### 1. Setup Project

Create a new project using this template:

```bash
# Option 1: Use as GitHub template (recommended)
# Click "Use this template" on GitHub

# Option 2: Clone and setup manually
git clone <your-repo-url>
cd your-project-name

# Option 3: Merge updates from this template
git remote add upstream https://github.com/mjun0812/PyTorch-Project-Template.git
git fetch upstream main
git merge --allow-unrelated-histories --squash upstream/main
```

### 2. Environment Configuration

```bash
# Copy environment template
cp template.env .env
# Edit .env with your API keys and settings
```

Example `.env` configuration:

```bash
# Slack notifications (optional)
# You can use either SLACK_TOKEN or SLACK_WEBHOOK_URL
SLACK_TOKEN="xoxb-your-token"
SLACK_CHANNEL="#notifications"
SLACK_USERNAME="Training Bot"

# Alternative: Webhook URL (simpler setup)
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# MLflow tracking
MLFLOW_TRACKING_URI="./result/mlruns"  # or remote URI

# Weights & Biases (optional)
WANDB_API_KEY="your-wandb-key"
```

### 3. Installation

Choose your preferred installation method:

#### Option A: Local Development (Recommended)

```bash
# Install dependencies
uv sync

# Setup development environment
uv run pre-commit install

# Run training
uv run python train.py config/dummy.yaml
```

#### Option B: Docker

```bash
# Build container
./docker/build.sh

# Run training in container
./docker/run.sh python train.py config/dummy.yaml
```

#### Option C: Development Container

Open the project in VS Code and use the devcontainer configuration for a consistent development environment.

## Usage

### Basic Training

Start with the dummy configuration to test your setup:

```bash
# Basic training with dummy dataset
python train.py config/dummy.yaml

# Override configuration from command line
python train.py config/dummy.yaml batch=32 gpu.use=0 optimizer.lr=0.001
```

### Configuration Management

This template uses hierarchical configuration with inheritance support:

```bash
# Use dot notation to modify nested values
python train.py config/dummy.yaml gpu.use=0,1 model.backbone.depth=50

# Multiple overrides
python train.py config/dummy.yaml batch=64 epoch=100 optimizer.lr=0.01

# View current configuration
python script/show_config.py config/dummy.yaml

# Batch edit configuration files
python script/edit_configs.py config/dummy.yaml "optimizer.lr=0.01,batch=64"
```

Configuration hierarchy:

1. Dataclass defaults (`src/config/config.py`)
2. Base configs (`config/__base__/`)
3. Experiment configs (`config/*.yaml`) with `__base__` inheritance
4. CLI overrides

### Development Tools

```bash
# Launch Jupyter Lab for experimentation
./script/run_notebook.sh

# Start MLflow UI for experiment tracking
./script/run_mlflow.sh

# View all registered components
python script/show_registers.py

# View model architecture
python script/show_model.py

# Visualize learning rate schedules
python script/show_scheduler.py

# View data transformation pipeline
python script/show_transform.py

# Clean up orphaned result directories
python script/clean_result.py

# Aggregate MLflow results to CSV
python script/aggregate_mlflow.py all

# Start documentation server (auto-reloads on changes)
./script/run_docs.sh
```

### Distributed Training

Scale your training across multiple GPUs and nodes:

#### Single Node, Multiple GPUs

```bash
# Use torchrun for DDP training (recommended)
./torchrun.sh 4 train.py config/dummy.yaml gpu.use="0,1,2,3"

# Alternative: DataParallel (not recommended for production)
python train.py config/dummy.yaml gpu.use="0,1,2,3" gpu.multi_strategy="dp"
```

#### Multi-Node Training

```bash
# Master node (node 0)
./multinode.sh 2 4 12345 0 master-ip:12345 train.py config/dummy.yaml gpu.use="0,1,2,3"

# Worker nodes (node 1+)
./multinode.sh 2 4 12345 1 master-ip:12345 train.py config/dummy.yaml gpu.use="0,1,2,3"
```

#### FSDP (Fully Sharded Data Parallel)

For very large models that don't fit in GPU memory:

```bash
python train.py config/dummy.yaml gpu.multi_strategy="fsdp" gpu.fsdp.min_num_params=100000000
```

### Results and Checkpointing

Training results are automatically saved to:

```
result/[dataset_name]/[date]_[model_name]_[tag]/
├── config.yaml          # Complete configuration used
├── models/              # Model checkpoints (latest.pth, best.pth, epoch_N.pth)
├── optimizers/          # Optimizer states  
└── schedulers/          # Scheduler states
```

### Resume Training

Resume interrupted training using saved checkpoints:

```bash
# Resume from automatically saved checkpoint
python train.py result/dataset_name/20240108_ResNet_experiment/config.yaml

# Resume and extend training
python train.py result/dataset_name/20240108_ResNet_experiment/config.yaml epoch=200

# Resume with different configuration
python train.py result/dataset_name/20240108_ResNet_experiment/config.yaml gpu.use=1 batch=64
```

### Evaluation

Run evaluation separately from training:

```bash
# Evaluate using saved model configuration
python test.py result/dataset_name/20240108_ResNet_experiment/config.yaml

# Evaluate with different GPU
python test.py result/dataset_name/20240108_ResNet_experiment/config.yaml gpu.use=1
```

### Performance Optimization

#### RAM Caching

Speed up training by caching datasets in RAM:

```bash
python train.py config/dummy.yaml use_ram_cache=true ram_cache_size_gb=16
```

Implement caching in your custom dataset:

```python
if self.cache is not None and idx in self.cache:
    data = self.cache.get(idx)
else:
    data = self.load_data(idx)  # Your data loading logic
    if self.cache is not None:
        self.cache.set(idx, data)
```

#### Mixed Precision Training

```bash
# Enable automatic mixed precision with fp16
python train.py config/dummy.yaml use_amp=true amp_dtype="fp16"

# Use bfloat16 for newer hardware (A100, H100)
python train.py config/dummy.yaml use_amp=true amp_dtype="bf16"
```

#### torch.compile

```bash
# Enable PyTorch 2.0 compilation for speedup
python train.py config/dummy.yaml use_compile=true compile_backend="inductor"

# Alternative backends
python train.py config/dummy.yaml use_compile=true compile_backend="aot_eager"
```

### Slack Notifications

Get notified about training progress and errors:

```bash
# Training will automatically send notifications on completion/error
python train.py config/dummy.yaml

# Manual notification testing
uv run --frozen pytest tests/test_slack_notification.py -v
```

## Architecture

### Project Structure

```
src/
├── config/          # Configuration management with inheritance
├── dataloaders/     # Dataset and DataLoader implementations  
├── models/          # Model definitions and backbones
│   ├── backbone/    # Pre-trained backbones (ResNet, Swin, etc.)
│   ├── layers/      # Custom layers and building blocks
│   └── losses/      # Loss function implementations
├── optimizer/       # Optimizer builders (including ScheduleFree)
├── scheduler/       # Learning rate schedulers
├── transform/       # Data preprocessing and augmentation
├── evaluator/       # Metrics and evaluation
├── runner/          # Training and testing loops
└── utils/           # Utilities (logger, registry, torch utils)

config/
├── __base__/        # Base configuration templates
└── *.yaml          # Experiment configurations

script/              # Utility scripts
├── run_*.sh         # Service startup scripts
├── show_*.py        # Visualization tools
└── aggregate_*.py   # Result aggregation tools
```

### Registry System

Components are registered using decorators for dynamic instantiation:

```python
from src.models import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class MyModel(BaseModel):
    def __init__(self, ...):
        super().__init__()
        # Model implementation

# Custom name registration
@MODEL_REGISTRY.register("custom_name")
class AnotherModel(BaseModel):
    pass
```

Available registries:

- `MODEL_REGISTRY`: Model architectures
- `DATASET_REGISTRY`: Dataset implementations
- `TRANSFORM_REGISTRY`: Data transformations
- `OPTIMIZER_REGISTRY`: Optimizers
- `LR_SCHEDULER_REGISTRY`: Learning rate schedulers
- `EVALUATOR_REGISTRY`: Evaluation metrics

### Configuration System

The configuration system supports inheritance and modular composition:

```yaml
# config/my_experiment.yaml
__base__: "__base__/config.yaml"

# Override specific values
batch: 64
optimizer:
  lr: 0.001
  
# Import specific sections
transform:
  __import__: "__base__/transform/imagenet.yaml"
```

### Error Handling and Notifications

The template includes comprehensive error handling:

- **Automatic Slack notifications** for training completion and errors
- **Graceful error recovery** with detailed logging
- **Checkpoint preservation** even during failures
- **Distributed training fault tolerance**

## Development

### Testing

```bash
# Run all tests
uv run --frozen pytest

# Run specific test modules
uv run --frozen pytest tests/test_modules.py
uv run --frozen pytest tests/test_slack_notification.py -v

# Run with verbose output
uv run --frozen pytest -v
```

### Code Quality

```bash
# Format code
uv run --frozen ruff format .

# Check code quality
uv run --frozen ruff check .

# Fix auto-fixable issues
uv run --frozen ruff check . --fix
```

### Documentation

```bash
# Start documentation server with live reload
./script/run_docs.sh
```

### Docker Development

```bash
# Build development image
./docker/build.sh

# Run commands in container
./docker/run.sh python train.py config/dummy.yaml
./docker/run.sh bash  # Interactive shell
```
