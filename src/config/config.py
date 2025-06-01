from dataclasses import dataclass, field


@dataclass
class BaseClassConfig:
    """Base config class for creating a class instance"""

    name: str = ""
    class_name: str = ""
    args: dict | None = None


@dataclass
class OptimizerGroupConfig:
    name: str
    divide: float


@dataclass
class OptimizerConfig(BaseClassConfig):
    checkpoint: str | None = None
    lr: float = 1e-3
    group: list[OptimizerGroupConfig] | None = None


@dataclass
class LrSchedulerConfig(BaseClassConfig):
    checkpoint: str | None = None


@dataclass
class LrSchedulersConfig:
    iter_scheduler: LrSchedulerConfig | None = None
    epoch_scheduler: LrSchedulerConfig | None = None


@dataclass
class EvaluatorConfig(BaseClassConfig):
    pass


@dataclass
class EvaluatorsConfig:
    train: list[EvaluatorConfig] | None = None
    val: list[EvaluatorConfig] | None = None
    test: list[EvaluatorConfig] | None = None


@dataclass
class TransformConfig(BaseClassConfig):
    pass


@dataclass
class DatasetConfig(BaseClassConfig):
    transforms: list[TransformConfig] = field(default_factory=list)
    batch_transforms: list[TransformConfig] | None = None


@dataclass
class DatasetsConfig:
    train: DatasetConfig | None = None
    val: DatasetConfig | None = None
    test: DatasetConfig | None = None
    batch_sampler: str | None = None


@dataclass
class LossConfig(BaseClassConfig):
    pass


@dataclass
class ModelConfig(BaseClassConfig):
    checkpoint: str | None = None
    pre_trained_weight: str | None = None
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
    mlflow_ignore_artifact_dirs: list[str] = field(
        default_factory=lambda: ["models", "optimizers", "schedulers"]
    )
    log_params: list[LogParamsConfig] | None = None


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
    best_metric: float = 0
    best_epoch: int = 0

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
    amp_init_scale: float | None = 2**16

    # torch.compile
    use_compile: bool = False
    compile_backend: str = "inductor"

    use_clip_grad: bool = True
    clip_grad_norm: float = 10
    gradient_accumulation_steps: int = 1

    output: str = "./result"
    tag: str | None = None

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetsConfig = field(default_factory=DatasetsConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LrSchedulersConfig = field(default_factory=LrSchedulersConfig)
    evaluator: EvaluatorsConfig = field(default_factory=EvaluatorsConfig)

    log: LoggerConfig = field(default_factory=LoggerConfig)
