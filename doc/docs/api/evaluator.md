# Evaluator API Reference

Evaluation metrics and validation framework for model assessment.

## Evaluator Builder

::: src.evaluator.build
    options:
      show_source: true
      show_root_heading: true

## Base Classes

::: src.evaluator.base
    options:
      show_source: true
      show_root_heading: true

## Usage Examples

### Building an Evaluator

```python
from src.evaluator.build import build_evaluator
from omegaconf import DictConfig

# Configuration for evaluator
evaluator_config = DictConfig({
    "name": "dummy_evaluator",
    "metrics": ["accuracy", "f1_score", "precision", "recall"],
    "average": "macro"
})

evaluator = build_evaluator(evaluator_config)
```

### Custom Evaluator Registration

```python
from src.evaluator import EVALUATOR_REGISTRY
from src.evaluator.base import BaseEvaluator
import torch

@EVALUATOR_REGISTRY.register("my_custom_evaluator")
class MyCustomEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.metrics = config.metrics
        
    def evaluate(self, predictions, targets):
        """
        Evaluate predictions against targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of metric results
        """
        results = {}
        
        # Compute accuracy
        if "accuracy" in self.metrics:
            correct = (predictions.argmax(dim=1) == targets).float()
            results["accuracy"] = correct.mean().item()
            
        # Compute custom metrics
        if "custom_metric" in self.metrics:
            results["custom_metric"] = self.compute_custom_metric(predictions, targets)
            
        return results
        
    def compute_custom_metric(self, predictions, targets):
        # Implement custom metric computation
        return metric_value
```

### Using Evaluators in Training

```python
from src.trainer import Trainer
from src.evaluator.build import build_evaluator

# Setup evaluator
evaluator_config = DictConfig({
    "name": "classification_evaluator",
    "metrics": ["accuracy", "f1_score"],
    "num_classes": 10
})

evaluator = build_evaluator(evaluator_config)

# Use in training loop
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    evaluator=evaluator,
    config=config
)

trainer.train()
```

### Evaluation Workflow

```python
import torch

# Collect predictions and targets
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch in val_loader:
        inputs, targets = batch
        predictions = model(inputs)
        
        all_predictions.append(predictions)
        all_targets.append(targets)

# Concatenate all results
predictions = torch.cat(all_predictions, dim=0)
targets = torch.cat(all_targets, dim=0)

# Evaluate
results = evaluator.evaluate(predictions, targets)
print(f"Validation Results: {results}")
```

### Multi-Task Evaluation

```python
# Configuration for multi-task evaluation
multitask_config = DictConfig({
    "name": "multitask_evaluator",
    "tasks": {
        "classification": {
            "metrics": ["accuracy", "f1_score"],
            "num_classes": 10
        },
        "regression": {
            "metrics": ["mse", "mae"],
            "loss_type": "mse"
        }
    }
})

multitask_evaluator = build_evaluator(multitask_config)

# Evaluate multiple tasks
task_predictions = {
    "classification": class_predictions,
    "regression": reg_predictions
}
task_targets = {
    "classification": class_targets,
    "regression": reg_targets
}

results = multitask_evaluator.evaluate(task_predictions, task_targets)
```

### Available Metrics

Common metrics available through the evaluation framework:

#### Classification Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score (macro/micro/weighted average)
- **Recall**: Recall score (macro/micro/weighted average)
- **F1-Score**: F1 score (macro/micro/weighted average)
- **AUC-ROC**: Area under ROC curve
- **Confusion Matrix**: Detailed confusion matrix

#### Regression Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination

#### Custom Metrics

- Implement domain-specific metrics by extending BaseEvaluator
- Support for batch-wise and epoch-wise aggregation
- Configurable metric computation and reporting

### Integration with MLflow

```python
import mlflow

# Log evaluation results
results = evaluator.evaluate(predictions, targets)

for metric_name, metric_value in results.items():
    mlflow.log_metric(f"val_{metric_name}", metric_value)
```
