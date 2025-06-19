# PyTorch Project Template

A comprehensive, production-ready PyTorch project template with modular architecture, distributed training support, and modern tooling.

## Features

- **Modular Architecture**: Registry-based component system for easy extensibility
- **Configuration Management**: OmegaConf + dataclasses with CLI override support  
- **Distributed Training**: Single-node/multi-node training with DDP, FSDP, and DataParallel
- **Experiment Tracking**: MLflow and Weights & Biases integration
- **Modern Tooling**: uv package management, pre-commit hooks, Docker support
- **Resume Training**: Automatic checkpoint saving and loading
- **Cross-Platform**: Development support on macOS and Linux
- **Development Environment**: Devcontainer and Jupyter Lab integration
- **RAM Caching**: Optional dataset caching for faster training
- **Documentation**: Auto-generated API docs with MkDocs and live reloading

## Requirements

- **Python**: 3.11+
- **Package Manager**: uv
- **CUDA**: 12.8
- **PyTorch**: 2.7.1

## Quick Start

### 1. Use this Template

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

Configure environment variables:

```bash
cp template.env .env
# Edit .env with your API keys and settings
```

Example `.env` configuration:

```bash
# Slack notifications (optional)
# You can use either SLACK_TOKEN or SLACK_WEBHOOK_URL
# If you set both, SLACK_TOKEN will be used
SLACK_TOKEN="your-slack-token"
SLACK_CHANNEL="your-slack-channel"
SLACK_USERNAME="your-slack-username"

SLACK_WEBHOOK_URL="your-slack-webhook-url"

# MLflow tracking
MLFLOW_TRACKING_URI="./result/mlruns"  # or remote URI
# MLFLOW_TRACKING_USERNAME=""  # for remote MLflow
# MLFLOW_TRACKING_PASSWORD=""

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

This template uses hierarchical configuration with OmegaConf and dataclasses:

```bash
# Use dot notation to modify nested values
python train.py config/dummy.yaml gpu.use=0,1 model.backbone.depth=50

# Multiple overrides
python train.py config/dummy.yaml batch=64 epoch=100 optimizer.lr=0.01
```

Configuration files are located in `config/` with base configurations in `config/__base__/`.

### Development Tools

```bash
# Launch Jupyter Lab for experimentation
./script/run_notebook.sh

# Start MLflow UI for experiment tracking
./script/run_mlflow.sh

# View all registered components
python script/show_registers.py

# Batch edit configuration files
python script/edit_configs.py config/dummy.yaml "optimizer.lr=0.01,batch=64"

# Clean up orphaned result directories
python script/clean_result.py | xargs -I{} -P 2 rm -rf {}

# Aggregate MLflow results
python script/aggregate_mlflow.py all

# Start documentation server (auto-reloads on changes)
./script/run_docs.sh
```

### Distributed Training

Scale your training across multiple GPUs and nodes:

#### Single Node, Multiple GPUs

```bash
# Use torchrun for DDP training
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

```bash
# For very large models
python train.py config/dummy.yaml gpu.multi_strategy="fsdp" gpu.fsdp.min_num_params=100000000
```

### Results and Checkpointing

Training results are automatically saved to:

```
result/[dataset_name]/[date]_[model_name]_[tag]/
├── config.yaml          # Complete configuration used
├── models/              # Model checkpoints
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
# Enable automatic mixed precision
python train.py config/dummy.yaml use_amp=true amp_dtype="fp16"

# Use bfloat16 for newer hardware
python train.py config/dummy.yaml use_amp=true amp_dtype="bf16"
```

#### torch.compile

```bash
# Enable PyTorch 2.0 compilation
python train.py config/dummy.yaml use_compile=true compile_backend="inductor"
```

## Architecture

### Project Structure

```
src/
├── config/          # Configuration management
├── dataloaders/     # Dataset and DataLoader implementations  
├── models/          # Model definitions and backbones
├── optimizer/       # Optimizer builders
├── scheduler/       # Learning rate schedulers
├── transform/       # Data preprocessing and augmentation
├── evaluator/       # Metrics and evaluation
├── trainer.py       # Training loop implementation
└── utils/           # Utilities and helpers

config/
├── __base__/        # Base configuration templates
└── *.yaml          # Experiment configurations
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
```

### Configuration Flow

1. Load base configuration from `config/__base__/config.yaml`
2. Merge with experiment-specific YAML file
3. Apply CLI overrides
4. Instantiate components using registry system
