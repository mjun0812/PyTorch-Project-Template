# Getting Started

This tutorial will guide you through setting up and running your first experiment with the PyTorch Project Template.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pytorch-project-template.git
cd pytorch-project-template
```

### 2. Install Dependencies

The project uses `uv` for package management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### 3. Setup Development Environment

```bash
# Install pre-commit hooks
uv run pre-commit install

# Verify installation
uv run python test.py
```

## Your First Experiment

### 1. Run the Dummy Example

Start with the built-in dummy configuration:

```bash
python train.py config/dummy.yaml
```

This will:
- Load the dummy dataset
- Initialize a simple model
- Train for a few epochs
- Save results to `result/dummy_dataset/`

### 2. Understanding the Output

The training script will create a directory structure like:

```
result/dummy_dataset/20241208_093000_dummy_tag/
├── config.yaml          # Full configuration used
├── checkpoints/          # Model checkpoints
│   ├── best.pth
│   └── last.pth
├── logs/                 # Training logs
│   └── training.log
└── mlruns/              # MLflow experiment tracking
```

### 3. Monitor Training

The template integrates with MLflow for experiment tracking:

```bash
# Start MLflow UI (in another terminal)
./script/run_mlflow.sh

# Open browser to http://localhost:5000
```

## Configuration Basics

### Understanding Configuration Files

The project uses hierarchical YAML configuration:

```yaml
# config/dummy.yaml
dataset: dummy_dataset
model: dummy
batch: 32
epoch: 10

optimizer:
  name: AdamW
  lr: 0.001
  weight_decay: 0.01

lr_scheduler:
  name: CosineAnnealingWarmRestarts
  T_0: 5
  T_mult: 2
```

### Overriding Configuration

You can override any configuration value from the command line:

```bash
# Change batch size and learning rate
python train.py config/dummy.yaml batch=64 optimizer.lr=0.01

# Change model and training epochs
python train.py config/dummy.yaml model=resnet50 epoch=100
```

## Multi-GPU Training

### Single Node, Multiple GPUs

```bash
# Train on GPUs 0 and 1
python train.py config/dummy.yaml gpu.use="0,1"

# Or use torchrun for better performance
./torchrun.sh 2 train.py config/dummy.yaml gpu.use="0,1"
```

### Multi-Node Training

```bash
# On each node, specify the master address
export MASTER_ADDR="192.168.1.100"
export MASTER_PORT="29500"

# Node 0 (rank 0-1)
./torchrun.sh 2 train.py config/dummy.yaml gpu.use="0,1" --node_rank=0 --nnodes=2

# Node 1 (rank 2-3)
./torchrun.sh 2 train.py config/dummy.yaml gpu.use="0,1" --node_rank=1 --nnodes=2
```

## Common Commands

### Development

```bash
# Run tests
uv run --frozen pytest

# Format code
uv run --frozen ruff format .

# Check code quality
uv run --frozen ruff check .

# View registered components
python script/show_registers.py
```

### Training Variants

```bash
# Resume from checkpoint
python train.py result/dummy_dataset/20241208_093000_dummy_tag/config.yaml

# Train with mixed precision
python train.py config/dummy.yaml gpu.mixed_precision=true

# Debug mode (single batch)
python train.py config/dummy.yaml debug=true
```

### Utilities

```bash
# Clean up old results
python script/clean_result.py --days 7

# Show model architecture
python script/show_model.py config/dummy.yaml

# Edit multiple configs at once
python script/edit_configs.py config/dummy.yaml "optimizer.lr=0.01,batch=64"
```

## Next Steps

Now that you have the basic setup working:

1. **[Training Guide](training.md)**: Learn about advanced training features
2. **[Custom Components](custom-components.md)**: Add your own models and datasets
3. **[API Reference](../api/config.md)**: Explore the detailed API documentation

## Troubleshooting

### Common Issues

**CUDA out of memory**
```bash
# Reduce batch size
python train.py config/dummy.yaml batch=16

# Enable gradient checkpointing (if supported by model)
python train.py config/dummy.yaml model.gradient_checkpointing=true
```

**Import errors**
```bash
# Ensure you're in the project root and using uv
cd pytorch-project-template
uv run python train.py config/dummy.yaml
```

**Configuration errors**
```bash
# Validate configuration
python script/show_config.py config/dummy.yaml
```

For more troubleshooting tips, check the project's issue tracker or documentation.