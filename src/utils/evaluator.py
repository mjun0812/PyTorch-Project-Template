import torchmetrics


def build_evaluator(cfg, phase="train"):
    metric_fn = torchmetrics.MetricCollection(
        [
            torchmetrics.Accuracy(task="multiclass", num_classes=cfg.DATASET.NUM_CLASSES),
            torchmetrics.Precision(
                task="multiclass", num_classes=cfg.DATASET.NUM_CLASSES, average="macro"
            ),
            torchmetrics.Recall(
                task="multiclass", num_classes=cfg.DATASET.NUM_CLASSES, average="macro"
            ),
            torchmetrics.F1Score(
                task="multiclass", num_classes=cfg.DATASET.NUM_CLASSES, average="macro"
            ),
        ]
    )
    if cfg.DATASET.NUM_CLASSES >= 5 and phase == "test":
        metric_fn.add_metrics(
            {
                "Accuracy@5": torchmetrics.Accuracy(
                    task="multiclass", top_k=5, num_classes=cfg.DATASET.NUM_CLASSES
                )
            }
        )
    return metric_fn
