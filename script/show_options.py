from src.dataloaders import DATASET_REGISTRY
from src.evaluator import EVALUATOR_REGISTRY
from src.losses import LOSS_REGISTRY
from src.models import MODEL_REGISTRY
from src.optimizer import OPTIMIZER_REGISTRY
from src.scheduler import SCHEDULER_REGISTRY


def main():
    print(DATASET_REGISTRY)
    print(MODEL_REGISTRY)
    print(LOSS_REGISTRY)
    print(EVALUATOR_REGISTRY)
    print(OPTIMIZER_REGISTRY)
    print(SCHEDULER_REGISTRY)


if __name__ == "__main__":
    main()
