from pprint import pprint

from src.dataloaders import DATASET_REGISTRY
from src.evaluator import EVALUATOR_REGISTRY
from src.models import MODEL_REGISTRY
from src.models.backbone import get_available_backbones
from src.optimizer import OPTIMIZER_REGISTRY
from src.scheduler import SCHEDULER_REGISTRY
from src.transform import BATCHED_TRANSFORM_REGISTRY, TRANSFORM_REGISTRY


def main() -> None:
    blue = "\033[94m"
    reset = "\033[0m"

    print(f"{blue}DATASET_REGISTRY{reset}")
    print(DATASET_REGISTRY)

    print(f"{blue}EVALUATOR_REGISTRY{reset}")
    print(EVALUATOR_REGISTRY)

    print(f"{blue}MODEL_REGISTRY{reset}")
    print(MODEL_REGISTRY)

    print(f"{blue}BACKBONE_REGISTRY{reset}")
    pprint(get_available_backbones())

    print(f"{blue}OPTIMIZER_REGISTRY{reset}")
    print(OPTIMIZER_REGISTRY)

    print(f"{blue}SCHEDULER_REGISTRY{reset}")
    print(SCHEDULER_REGISTRY)

    print(f"{blue}TRANSFORM_REGISTRY{reset}")
    print(TRANSFORM_REGISTRY)

    print(f"{blue}BATCHED_TRANSFORM_REGISTRY{reset}")
    print(BATCHED_TRANSFORM_REGISTRY)


if __name__ == "__main__":
    main()
