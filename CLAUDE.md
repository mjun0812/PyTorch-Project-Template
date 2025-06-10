# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This document contains critical information about working with this PyTorch project template.
Follow these guidelines precisely.

## Rules

1. Package Management
   - ONLY use uv, NEVER pip
   - Installation: `uv add package`
   - Upgrading: `uv add --dev package --upgrade-package package`
   - FORBIDDEN: `uv pip install`, `@latest` syntax

2. Code Quality
   - Type hints required for all code
   - Follow existing patterns exactly
   - Use Google style for docstring

3. Testing Requirements
   - Framework: `uv run --frozen pytest`
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests

4. Git
   - Follow the Conventional Commits style on commit messages.

## Code Formatting and Linting

1. Ruff
   - Format: `uv run --frozen ruff format .`
   - Check: `uv run --frozen ruff check .`
   - Fix: `uv run --frozen ruff check . --fix`
2. Pre-commit
   - Config: `.pre-commit-config.yaml`
   - Runs: on git commit
   - Tools: Ruff (Python)

## Project Architecture

This is a PyTorch project template with modular components connected through a registry system:

### Core Design Patterns

- **Registry System**: All modules use `@REGISTRY.register()` decorators for dynamic component discovery. Registry objects in `src/utils/registry.py` provide name-to-class mapping
- **Config Management**: `ConfigManager` in `src/config/manager.py` merges dataclass defaults → YAML files → CLI overrides. Supports `__base__` and `__import__` keys for config inheritance
- **Build Pattern**: Each component type has a `build.py` module that instantiates objects from config using registries. Pattern: `build_*()` functions take config and return instantiated objects

### Component Registration

All components must be registered to be discoverable:

```python
from src.models import MODEL_REGISTRY

@MODEL_REGISTRY.register()  # Auto-registers with class name
class MyModel(nn.Module): ...

@MODEL_REGISTRY.register("custom_name")  # Registers with custom name
class AnotherModel(nn.Module): ...
```

### Configuration Hierarchy

1. **Dataclass Defaults**: `src/config/config.py` defines `ExperimentConfig` with type-safe defaults
2. **Base Configs**: `config/__base__/` contains component-specific YAML templates
3. **Experiment Configs**: `config/*.yaml` files inherit from base configs using `__base__` key
4. **CLI Overrides**: Dot notation supported: `optimizer.lr=0.001`

### Key Modules

- `src/config/`: Configuration management with `ConfigManager` and dataclass definitions
- `src/models/`: Model definitions with backbone support (ResNet, Swin, InternImage)
- `src/dataloaders/`: Dataset and DataLoader builders with RAM caching support
- `src/trainer.py`: Main training loop with DDP/FSDP support
- `src/optimizer/`: Optimizer builders including custom ones (Lion, ScheduleFree)
- `src/scheduler/`: Learning rate scheduler builders
- `src/transform/`: Data augmentation and preprocessing pipeline
- `src/evaluator/`: Metric evaluation framework

### Training Flow

1. `train.py` → `ConfigManager.build()` → loads config from YAML + CLI args
2. Build components using registry pattern: `build_model(cfg.model)`, `build_dataset(cfg.dataset)`
3. `Trainer` handles epoch/iteration loops, checkpointing, distributed training
4. Results saved to `result/[dataset]/[date]_[model]_[tag]/`

### Slack Notifications

The project includes built-in Slack notification support for training completion and errors:

- **Environment Variables**: Set `SLACK_TOKEN` or `SLACK_WEBHOOK_URL` in `.env`
- **Channel Override**: Use `SLACK_CHANNEL` to set default notification channel
- **Username Override**: Use `SLACK_USERNAME` to set default notification username
- **Priority**: Webhook URL > Token, Environment variables > function arguments
- **Auto-notifications**: Training completion, errors, and test results sent automatically

Configuration in `.env`:

```bash
# Option 1: Slack API Token (recommended)
SLACK_TOKEN="xoxb-your-token"
SLACK_CHANNEL="#notifications"  # Optional: default channel
SLACK_USERNAME="Bot Name"       # Optional: default username

# Option 2: Webhook URL (simpler setup)
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

## Common Commands

### Training

```bash
# Basic training
python train.py config/dummy.yaml

# With config overrides (dot notation for nested values)
python train.py config/dummy.yaml batch=32 gpu.use=0,1 optimizer.lr=0.001

# Multi-GPU training (DDP recommended)
./torchrun.sh 4 train.py config/dummy.yaml gpu.use="0,1,2,3"

# FSDP for very large models
python train.py config/dummy.yaml gpu.multi_strategy="fsdp" gpu.fsdp.min_num_params=100000000

# Resume training from checkpoint
python train.py result/dataset_name/timestamp_model_tag/config.yaml epoch=200

# Multi-node training (run on each node)
./multinode.sh 2 4 12345 0 master-ip:12345 train.py config/dummy.yaml gpu.use="0,1,2,3"
```

### Testing/Evaluation

```bash
# Run evaluation
python test.py config/dummy.yaml

# Evaluate saved model
python test.py result/dataset_name/timestamp_model_tag/config.yaml

# Test with different GPU
python test.py result/dataset_name/timestamp_model_tag/config.yaml gpu.use=1
```

### Development

```bash
# Install dependencies
uv sync

# Setup pre-commit
uv run pre-commit install

# Run all tests
uv run --frozen pytest

# Run specific test function
uv run --frozen pytest tests/test_modules.py::test_config

# Run specific test file
uv run --frozen pytest tests/test_modules.py

# Run with verbose output
uv run --frozen pytest -v

# Show registered modules (useful for debugging)
python script/show_registers.py

# Run specific Slack notification tests
uv run --frozen pytest tests/test_slack_notification.py -v

# View model architecture  
python script/show_model.py

# View scheduler visualization
python script/show_scheduler.py

# View transform pipeline
python script/show_transform.py

# View current config
python script/show_config.py config/dummy.yaml

# Batch edit configs
python script/edit_configs.py config/dummy.yaml "optimizer.lr=0.01,batch=64"

# Clean orphaned results
python script/clean_result.py

# Aggregate MLflow results
python script/aggregate_mlflow.py all
```

### Docker

```bash
./docker/build.sh
./docker/run.sh python train.py config/dummy.yaml
```

### Tools and Environment

```bash
# Copy environment template
cp template.env .env
# Edit .env with your API keys and settings

# MLflow UI for experiment tracking
./script/run_mlflow.sh

# Jupyter Lab for experimentation
./script/run_notebook.sh

# Performance optimization options
python train.py config/dummy.yaml use_ram_cache=true ram_cache_size_gb=16
python train.py config/dummy.yaml use_amp=true amp_dtype="fp16"
python train.py config/dummy.yaml use_compile=true compile_backend="inductor"
```

### Documentation

```bash
# Start documentation server (auto-reloads on changes)
./script/run_docs.sh

# Build documentation
uv run mkdocs build

# Deploy to GitHub Pages (if configured)
uv run mkdocs gh-deploy
```

## Key Project Information

### Result Structure

Training outputs are saved to `result/[dataset]/[date]_[model]_[tag]/` containing:

- `config.yaml`: Complete configuration used for training
- `models/`: Model checkpoints (latest.pth, best.pth, epoch_N.pth)
- `optimizers/`: Optimizer state dictionaries
- `schedulers/`: Learning rate scheduler states

### Environment Variables (.env)

Required/Optional settings in `.env` file:

- `SLACK_TOKEN` or `SLACK_WEBHOOK_URL`: Slack notifications
- `SLACK_CHANNEL`, `SLACK_USERNAME`: Default Slack settings
- `MLFLOW_TRACKING_URI`: MLflow tracking server
- `WANDB_API_KEY`: Weights & Biases integration

### Registry System Usage

All components must be registered for discovery:

```python
from src.models import MODEL_REGISTRY

@MODEL_REGISTRY.register()  # Use class name
@MODEL_REGISTRY.register("custom_name")  # Use custom name
```

Available registries: `MODEL_REGISTRY`, `DATASET_REGISTRY`, `TRANSFORM_REGISTRY`, `OPTIMIZER_REGISTRY`, `LR_SCHEDULER_REGISTRY`, `EVALUATOR_REGISTRY`

## important-instruction-reminders

Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
