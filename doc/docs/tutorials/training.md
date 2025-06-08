# Training Guide

This guide covers advanced training features and best practices for the PyTorch Project Template.

## Training Configuration

### Basic Training Setup

```yaml
# config/my_experiment.yaml
dataset: my_dataset
model: resnet50
batch: 32
epoch: 100

optimizer:
  name: AdamW
  lr: 0.001
  weight_decay: 0.01

lr_scheduler:
  name: CosineAnnealingWarmRestarts
  T_0: 10
  T_mult: 2

gpu:
  use: "0"
  mixed_precision: true
```

### Advanced Training Options

```yaml
# Advanced configuration
training:
  gradient_clip_norm: 1.0
  accumulate_grad_batches: 4
  validation_frequency: 5
  save_frequency: 10
  early_stopping:
    patience: 20
    monitor: "val_loss"
    mode: "min"

checkpoint:
  save_best: true
  save_last: true
  save_top_k: 3
  monitor: "val_accuracy"
  mode: "max"
```

## Distributed Training

### Data Parallel (DP)

```bash
# Simple data parallel on multiple GPUs
python train.py config/my_experiment.yaml gpu.use="0,1,2,3"
```

### Distributed Data Parallel (DDP)

```bash
# More efficient DDP training
./torchrun.sh 4 train.py config/my_experiment.yaml gpu.use="0,1,2,3"
```

### Fully Sharded Data Parallel (FSDP)

```yaml
# Enable FSDP in configuration
distributed:
  strategy: "fsdp"
  precision: "bf16"
  sharding_strategy: "full_shard"
```

```bash
# Run with FSDP
./torchrun.sh 8 train.py config/my_experiment.yaml distributed.strategy=fsdp
```

## Mixed Precision Training

### Automatic Mixed Precision (AMP)

```yaml
gpu:
  mixed_precision: true
  precision: "16"  # or "bf16" for newer GPUs
```

### Custom Precision Settings

```python
# In model configuration
model:
  name: my_model
  precision: "bf16"
  compile: true  # PyTorch 2.0 compilation
```

## Gradient Accumulation

For training with larger effective batch sizes:

```yaml
training:
  accumulate_grad_batches: 4  # Effective batch = batch * accumulate_grad_batches
  gradient_clip_norm: 1.0     # Clip gradients to prevent explosion
```

## Checkpointing and Resuming

### Automatic Checkpointing

```yaml
checkpoint:
  save_frequency: 10        # Save every 10 epochs
  save_best: true          # Save best model based on metric
  save_last: true          # Always save latest checkpoint
  monitor: "val_accuracy"   # Metric to monitor for best model
  mode: "max"              # "max" for accuracy, "min" for loss
```

### Resuming Training

```bash
# Resume from specific checkpoint directory
python train.py result/my_dataset/20241208_093000_resnet50_experiment/config.yaml

# Resume from specific checkpoint file
python train.py config/my_experiment.yaml checkpoint.resume_from="path/to/checkpoint.pth"

# Resume and train for more epochs
python train.py result/my_dataset/20241208_093000_resnet50_experiment/config.yaml epoch=200
```

## Learning Rate Scheduling

### Common Schedulers

```yaml
# Cosine annealing with warm restarts
lr_scheduler:
  name: CosineAnnealingWarmRestarts
  T_0: 10
  T_mult: 2
  eta_min: 1e-6

# Step LR with decay
lr_scheduler:
  name: StepLR
  step_size: 30
  gamma: 0.1

# Reduce on plateau
lr_scheduler:
  name: ReduceLROnPlateau
  mode: "min"
  factor: 0.5
  patience: 10
```

### Custom Warmup Scheduling

```yaml
lr_scheduler:
  name: CosineAnnealingWarmupReduceRestarts
  T_0: 50
  T_mult: 2
  eta_min: 1e-6
  T_up: 10      # Warmup epochs
  gamma: 0.5    # Restart decay factor
```

## Optimization Strategies

### Optimizer Selection

```yaml
# AdamW (recommended for most cases)
optimizer:
  name: AdamW
  lr: 0.001
  weight_decay: 0.01
  betas: [0.9, 0.999]

# SGD with momentum (for some computer vision tasks)
optimizer:
  name: SGD
  lr: 0.1
  momentum: 0.9
  weight_decay: 1e-4

# Lion optimizer (new, efficient)
optimizer:
  name: Lion
  lr: 1e-4
  betas: [0.9, 0.99]
  weight_decay: 0.01
```

### Schedule-Free Optimizers

```yaml
# AdamW Schedule-Free (no LR scheduler needed)
optimizer:
  name: AdamWScheduleFree
  lr: 0.001
  betas: [0.9, 0.999]
  weight_decay: 0.01
  warmup_steps: 1000
```

## Monitoring and Logging

### MLflow Integration

```python
# Automatic MLflow logging is enabled by default
# Access logs at: http://localhost:5000 (after running ./script/run_mlflow.sh)
```

### Custom Metrics Logging

```python
# In custom evaluator
import mlflow

def log_custom_metrics(self, metrics_dict):
    for key, value in metrics_dict.items():
        mlflow.log_metric(key, value)
```

### TensorBoard Integration

```yaml
logging:
  tensorboard: true
  log_dir: "logs"
  log_frequency: 100  # Log every 100 steps
```

## Evaluation and Validation

### Validation Configuration

```yaml
evaluation:
  frequency: 5        # Validate every 5 epochs
  metrics: ["accuracy", "f1_score", "precision", "recall"]
  save_predictions: true
  
evaluator:
  name: classification_evaluator
  num_classes: 10
  average: "macro"
```

### Custom Evaluation

```python
@EVALUATOR_REGISTRY.register("my_evaluator")
class MyEvaluator(BaseEvaluator):
    def evaluate(self, predictions, targets):
        # Custom evaluation logic
        results = {}
        results["custom_metric"] = self.compute_custom_metric(predictions, targets)
        return results
```

## Performance Optimization

### Model Compilation (PyTorch 2.0+)

```yaml
model:
  compile: true
  compile_mode: "default"  # "default", "reduce-overhead", "max-autotune"
```

### Data Loading Optimization

```yaml
dataloader:
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
```

### Memory Optimization

```yaml
# Gradient checkpointing for large models
model:
  gradient_checkpointing: true

# Reduce precision
gpu:
  mixed_precision: true
  precision: "bf16"

# Smaller batch size with accumulation
batch: 16
training:
  accumulate_grad_batches: 4
```

## Debugging and Profiling

### Debug Mode

```bash
# Train on single batch for debugging
python train.py config/my_experiment.yaml debug=true
```

### Profiling

```yaml
profiling:
  enabled: true
  profile_memory: true
  record_shapes: true
  with_stack: true
```

### Common Issues and Solutions

**Out of Memory (OOM)**
```yaml
# Reduce batch size
batch: 16

# Enable gradient checkpointing
model:
  gradient_checkpointing: true

# Use gradient accumulation
training:
  accumulate_grad_batches: 4
```

**Slow Training**
```yaml
# Increase num_workers
dataloader:
  num_workers: 8

# Enable model compilation
model:
  compile: true

# Use mixed precision
gpu:
  mixed_precision: true
```

**NaN Loss/Gradients**
```yaml
# Gradient clipping
training:
  gradient_clip_norm: 1.0

# Lower learning rate
optimizer:
  lr: 1e-4

# Use more stable precision
gpu:
  precision: "32"
```

## Best Practices

1. **Start Small**: Begin with a small model and dataset to verify everything works
2. **Monitor Metrics**: Use MLflow or TensorBoard to track training progress
3. **Regular Checkpoints**: Save checkpoints frequently to avoid losing progress
4. **Validation**: Always validate on a held-out set to monitor overfitting
5. **Reproducibility**: Set random seeds and save full configurations
6. **Resource Monitoring**: Monitor GPU memory and utilization during training

## Example Training Scripts

### Basic Training

```bash
python train.py config/my_experiment.yaml
```

### Advanced Training with All Features

```bash
./torchrun.sh 4 train.py config/advanced.yaml \
  batch=64 \
  optimizer.lr=0.001 \
  gpu.mixed_precision=true \
  training.gradient_clip_norm=1.0 \
  checkpoint.save_frequency=5
```