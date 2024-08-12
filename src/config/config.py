from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BaseConfig:
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return getattr(self, key, default)


@dataclass
class LossConfig(BaseConfig):
    name: str = "BaseLoss"
    loss: str = "BaseLoss"
    args: dict = field(default_factory=dict)


@dataclass
class ModelConfig(BaseConfig):
    name: str = "BaseModel"
    model: str = "BaseModel"

    pre_trained_weight: Optional[str] = None
    trained_weight: Optional[str] = None

    use_sync_bn: bool = False
    find_unused_parameters: bool = False

    use_model_ema: bool = False
    model_ema_decay: float = 0.99

    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class OptimizerGroupConfig(BaseConfig):
    name: str
    divide: float


@dataclass
class OptimizerConfig(BaseConfig):
    name: str = "AdamW"
    optimizer: str = "AdamW"
    lr: float = 1e-3
    args: Optional[dict] = None
    checkpoint: Optional[str] = None
    group: Optional[list[OptimizerGroupConfig]] = None


@dataclass
class LrSchedulerConfig(BaseConfig):
    name: str = "StepLR"
    scheduler: str = "StepLR"
    args: Optional[dict] = None


@dataclass
class TransformConfig(BaseConfig):
    name: str = "BaseTransform"
    args: Optional[dict] = None


@dataclass
class DatasetConfig(BaseConfig):
    name: str = "BaseDataset"
    dataset: str = "BaseDataset"

    train_transforms: list[TransformConfig] = field(default_factory=list)
    train_batch_transforms: Optional[list[TransformConfig]] = None
    val_transforms: list[TransformConfig] = field(default_factory=list)
    test_transforms: list[TransformConfig] = field(default_factory=list)


@dataclass
class EvaluatorConfig(BaseConfig):
    name: str = "BaseEvaluator"
    args: Optional[dict] = None


@dataclass
class EvaluatorsConfig(BaseConfig):
    train: Optional[list[EvaluatorConfig]] = None
    val: Optional[list[EvaluatorConfig]] = None
    test: Optional[list[EvaluatorConfig]] = None


@dataclass
class MlflowLogParamsConfig(BaseConfig):
    name: str = "Model"
    value: str = "model.name"


@dataclass
class MlflowConfig(BaseConfig):
    use: bool = False
    experiment_name: str = "pytorch-project-template"
    ignore_artifact_dirs: list[str] = field(default_factory=lambda: ["models"])
    log_params: list[MlflowLogParamsConfig] = field(default_factory=list)


@dataclass
class GPUConfig(BaseConfig):
    use: str | int = 0
    multi: bool = False
    use_cudnn: bool = True


@dataclass
class ExperimentConfig(BaseConfig):
    epoch: int = 100
    last_epoch: int = 0
    val_interval: int = 5
    save_interval: int = 5

    use_iter_loop: bool = False
    step_iter: int = 1000
    max_iter: int = 10000

    batch: int = 32
    num_worker: int = 4
    use_ram_cache: bool = True

    seed: int = 42

    # device config
    gpu: GPUConfig = field(default_factory=GPUConfig)
    use_cpu: bool = False

    use_amp: bool = False
    amp_dtype: str = "fp16"
    amp_init_scale: Optional[float] = None

    # torch.compile
    use_compile: bool = False
    compile_backend: str = "inductor"

    use_clip_grad: bool = True
    clip_grad_norm: float = 10

    adjust_lr: bool = False

    output: str = "./result"
    tag: Optional[str] = None

    mlflow: MlflowConfig = field(default_factory=MlflowConfig)

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LrSchedulerConfig = field(default_factory=LrSchedulerConfig)

    train_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    val_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    test_dataset: DatasetConfig = field(default_factory=DatasetConfig)

    evaluator: EvaluatorsConfig = field(default_factory=EvaluatorsConfig)
