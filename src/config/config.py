from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseClassConfig:
    """Base config class for creating a class instance"""

    name: str = ""
    class_name: str = ""
    args: Optional[dict] = None


@dataclass
class OptimizerGroupConfig:
    name: str
    divide: float


@dataclass
class OptimizerConfig(BaseClassConfig):
    checkpoint: Optional[str] = None
    lr: float = 1e-3
    group: Optional[list[OptimizerGroupConfig]] = None


@dataclass
class LrSchedulerConfig(BaseClassConfig):
    checkpoint: Optional[str] = None


@dataclass
class LrSchedulersConfig:
    iter_scheduler: Optional[LrSchedulerConfig] = None
    epoch_scheduler: Optional[LrSchedulerConfig] = None


@dataclass
class EvaluatorConfig(BaseClassConfig):
    pass


@dataclass
class EvaluatorsConfig:
    train: Optional[list[EvaluatorConfig]] = None
    val: Optional[list[EvaluatorConfig]] = None
    test: Optional[list[EvaluatorConfig]] = None


@dataclass
class TransformConfig(BaseClassConfig):
    pass


@dataclass
class DatasetConfig(BaseClassConfig):
    transforms: list[TransformConfig] = field(default_factory=list)
    batch_transforms: Optional[list[TransformConfig]] = None


@dataclass
class DatasetsConfig:
    train: Optional[DatasetConfig] = None
    val: Optional[DatasetConfig] = None
    test: Optional[DatasetConfig] = None
    batch_sampler: Optional[str] = None


@dataclass
class LossConfig(BaseClassConfig):
    pass


@dataclass
class ModelConfig(BaseClassConfig):
    checkpoint: Optional[str] = None
    pre_trained_weight: Optional[str] = None
    use_sync_bn: bool = False
    find_unused_parameters: bool = False

    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class FsdpConfig:
    min_num_params: int = 100000000
    use_cpu_offload: bool = False


@dataclass
class GPUConfig:
    use: str | int = 0
    multi: bool = False
    use_cudnn: bool = True
    use_tf32: bool = False
    # dp, ddp, fsdp
    multi_strategy: str = "ddp"
    fsdp: FsdpConfig = field(default_factory=FsdpConfig)


@dataclass
class LogParamsConfig:
    name: str = "Model"
    value: str = "model.name"


@dataclass
class LoggerConfig:
    use_mlflow: bool = False
    use_wandb: bool = False
    wandb_project_name: str = "pytorch-project-template"
    mlflow_experiment_name: str = "pytorch-project-template"
    mlflow_ignore_artifact_dirs: list[str] = field(default_factory=lambda: ["models"])
    log_params: Optional[list[LogParamsConfig]] = None


@dataclass
class ExperimentConfig:
    # Epoch Based Training
    epoch: int = 100
    last_epoch: int = 0

    # Iteration Based Training
    use_iter_loop: bool = False
    step_iter: int = 1000
    max_iter: int = 10000

    # Validation and Save Config
    val_interval: int = 5
    save_interval: int = 5
    metric_for_best_model: str = "total_loss"
    greater_is_better: bool = False

    # DataLoader Config
    batch: int = 32
    num_worker: int = 4
    use_ram_cache: bool = True
    ram_cache_size_gb: int = 16

    seed: int = 42

    # device config
    gpu: GPUConfig = field(default_factory=GPUConfig)
    use_cpu: bool = False

    # AutoMixedPrecision
    use_amp: bool = False
    amp_dtype: str = "fp16"
    amp_init_scale: Optional[float] = 2**16

    # torch.compile
    use_compile: bool = False
    compile_backend: str = "inductor"

    use_clip_grad: bool = True
    clip_grad_norm: float = 10
    gradient_accumulation_steps: int = 1

    output: str = "./result"
    tag: Optional[str] = None

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetsConfig = field(default_factory=DatasetsConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LrSchedulersConfig = field(default_factory=LrSchedulersConfig)
    evaluator: EvaluatorsConfig = field(default_factory=EvaluatorsConfig)

    log: LoggerConfig = field(default_factory=LoggerConfig)
