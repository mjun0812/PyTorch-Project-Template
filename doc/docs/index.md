# PyTorch Project Template

A modular PyTorch project template with configuration management and registry system.

## Overview

This project template provides a scalable and maintainable structure for PyTorch-based machine learning projects. It features:

- **Registry System**: Dynamic component registration and instantiation
- **Configuration Management**: OmegaConf + dataclasses for hierarchical configuration
- **Modular Architecture**: Pluggable components for models, datasets, optimizers, etc.
- **Distributed Training**: Support for DDP and FSDP
- **Development Tools**: Pre-commit hooks, testing, and documentation

## Key Features

### 🏗️ Architecture
- Modular design with clear separation of concerns
- Registry-based component system for easy extensibility
- Configuration-driven workflow with CLI override support

### 🚀 Training
- Multi-GPU and distributed training support
- Automatic mixed precision training
- Checkpointing and resuming
- MLflow integration for experiment tracking

### 🔧 Development
- Type hints throughout the codebase
- Comprehensive testing with pytest
- Code formatting with Ruff
- Pre-commit hooks for code quality

## Quick Start

```bash
# Install dependencies
uv sync

# Run basic training
python train.py config/dummy.yaml

# Multi-GPU training
./torchrun.sh 4 train.py config/dummy.yaml gpu.use="0,1,2,3"
```

## Documentation Structure

- **[Architecture](architecture/overview.md)**: Core concepts and design patterns
- **[API Reference](api/config.md)**: Detailed API documentation
- **[Tutorials](tutorials/getting-started.md)**: Step-by-step guides

## Project Structure

```
src/
├── config/          # Configuration management
├── models/          # Model definitions and builders
├── dataloaders/     # Dataset and DataLoader builders
├── optimizer/       # Optimizer builders
├── scheduler/       # Learning rate scheduler builders
├── transform/       # Data preprocessing and augmentation
├── evaluator/       # Evaluation metrics
└── utils/          # Utility functions
```