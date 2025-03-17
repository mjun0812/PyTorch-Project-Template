from dataclasses import dataclass, field
from typing import Optional

from .base import BaseConfig
from .config_dataset import DatasetConfig
from .config_model import ModelConfig


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
    checkpoint: Optional[str] = None
    args: Optional[dict] = None


@dataclass
class LrSchedulersConfig(BaseConfig):
    iter_scheduler: Optional[LrSchedulerConfig] = None
    epoch_scheduler: Optional[LrSchedulerConfig] = None


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
class LogParamsConfig(BaseConfig):
    name: str = "Model"
    value: str = "model.name"


@dataclass
class MlflowConfig(BaseConfig):
    use: bool = False
    experiment_name: str = "pytorch-project-template"
    ignore_artifact_dirs: list[str] = field(default_factory=lambda: ["models"])


@dataclass
class WandbConfig(BaseConfig):
    use: bool = False
    project_name: str = "pytorch-project-template"


@dataclass
class FsdpConfig(BaseConfig):
    min_num_params: int = 100000000
    use_cpu_offload: bool = False


@dataclass
class GPUConfig(BaseConfig):
    use: str | int = 0
    multi: bool = False
    use_cudnn: bool = True
    use_tf32: bool = False
    # dp, ddp, fsdp
    multi_strategy: str = "ddp"
    fsdp: FsdpConfig = field(default_factory=FsdpConfig)


@dataclass
class ExperimentConfig(BaseConfig):
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
    batch_sampler: Optional[str] = None
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

    adjust_lr: bool = False

    output: str = "./result"
    tag: Optional[str] = None

    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    log_params: list[LogParamsConfig] = field(default_factory=list)

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LrSchedulersConfig = field(default_factory=LrSchedulersConfig)

    # dataset
    train_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    val_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    test_dataset: DatasetConfig = field(default_factory=DatasetConfig)

    evaluator: EvaluatorsConfig = field(default_factory=EvaluatorsConfig)
