from dataclasses import dataclass, field


@dataclass
class BaseClassConfig:
    """Base config class for creating a class instance.

    This class provides common attributes needed for instantiating components
    through the registry system.

    Attributes:
        name: The name identifier for the component.
        class_name: The specific class name to instantiate.
        args: Additional arguments to pass to the component constructor.
    """

    name: str = ""
    class_name: str = ""
    args: dict | None = None


@dataclass
class OptimizerGroupConfig:
    """Configuration for optimizer parameter groups.

    Attributes:
        name: The name of the parameter group.
        divide: The division factor for learning rate scaling.
    """

    name: str
    divide: float


@dataclass
class OptimizerConfig(BaseClassConfig):
    """Configuration for optimizer settings.

    Attributes:
        checkpoint: Path to optimizer checkpoint for resuming training.
        lr: Learning rate for the optimizer.
        group: List of parameter groups with different settings.
    """

    checkpoint: str | None = None
    lr: float = 1e-3
    group: list[OptimizerGroupConfig] | None = None


@dataclass
class LrSchedulerConfig(BaseClassConfig):
    """Configuration for learning rate scheduler.

    Attributes:
        checkpoint: Path to scheduler checkpoint for resuming training.
    """

    checkpoint: str | None = None


@dataclass
class LrSchedulersConfig:
    """Configuration for multiple learning rate schedulers.

    Attributes:
        iter_scheduler: Scheduler applied per iteration.
        epoch_scheduler: Scheduler applied per epoch.
    """

    iter_scheduler: LrSchedulerConfig | None = None
    epoch_scheduler: LrSchedulerConfig | None = None


@dataclass
class EvaluatorConfig(BaseClassConfig):
    """Configuration for evaluator components.

    This class inherits all attributes from BaseClassConfig for component
    instantiation through the registry system.
    """

    pass


@dataclass
class EvaluatorsConfig:
    """Configuration for evaluators across different data splits.

    Attributes:
        train: List of evaluators for training data.
        val: List of evaluators for validation data.
        test: List of evaluators for test data.
    """

    train: list[EvaluatorConfig] | None = None
    val: list[EvaluatorConfig] | None = None
    test: list[EvaluatorConfig] | None = None


@dataclass
class TransformConfig(BaseClassConfig):
    """Configuration for data transformation components.

    This class inherits all attributes from BaseClassConfig for component
    instantiation through the registry system.
    """

    pass


@dataclass
class DatasetConfig(BaseClassConfig):
    """Configuration for dataset components.

    Attributes:
        transforms: List of transformations applied to individual samples.
        batch_transforms: List of transformations applied to batches.
    """

    transforms: list[TransformConfig] = field(default_factory=list)
    batch_transforms: list[TransformConfig] | None = None


@dataclass
class DatasetsConfig:
    """Configuration for datasets across different data splits.

    Attributes:
        train: Configuration for training dataset.
        val: Configuration for validation dataset.
        test: Configuration for test dataset.
        batch_sampler: Name of the batch sampler to use.
    """

    train: DatasetConfig | None = None
    val: DatasetConfig | None = None
    test: DatasetConfig | None = None
    batch_sampler: str | None = None


@dataclass
class LossConfig(BaseClassConfig):
    """Configuration for loss function components.

    This class inherits all attributes from BaseClassConfig for component
    instantiation through the registry system.
    """

    pass


@dataclass
class ModelConfig(BaseClassConfig):
    """Configuration for model components.

    Attributes:
        checkpoint: Path to model checkpoint for resuming training.
        pre_trained_weight: Path to pre-trained weights to load.
        use_sync_bn: Whether to use synchronized batch normalization.
        find_unused_parameters: Whether to find unused parameters in DDP.
        loss: Configuration for the loss function.
    """

    checkpoint: str | None = None
    pre_trained_weight: str | None = None
    use_sync_bn: bool = False
    find_unused_parameters: bool = False

    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class FsdpConfig:
    """Configuration for Fully Sharded Data Parallel (FSDP).

    Attributes:
        min_num_params: Minimum number of parameters to shard.
        use_cpu_offload: Whether to offload parameters to CPU.
    """

    min_num_params: int = 100000000
    use_cpu_offload: bool = False


@dataclass
class GPUConfig:
    """Configuration for GPU usage and distributed training.

    Attributes:
        device: Device type to use ("cuda", "mps", "cpu").
        use: GPU device(s) to use (string or int). Can be "mps" for MPS.
        multi: Whether to use multiple GPUs.
        use_cudnn: Whether to use cuDNN backend.
        use_tf32: Whether to use TensorFloat-32 format.
        multi_strategy: Strategy for multi-GPU training (dp, ddp, fsdp).
        fsdp: Configuration for FSDP when using fsdp strategy.
    """

    device: str = "cuda"
    use: str | int = 0
    multi: bool = False
    use_cudnn: bool = True
    use_tf32: bool = False
    # dp, ddp, fsdp
    multi_strategy: str = "ddp"
    fsdp: FsdpConfig = field(default_factory=FsdpConfig)


@dataclass
class LogParamsConfig:
    """Configuration for logging custom parameters.

    Attributes:
        name: Display name for the parameter.
        value: Dot-notation path to the parameter value.
    """

    name: str = "Model"
    value: str = "model.name"


@dataclass
class LoggerConfig:
    """Configuration for experiment logging.

    Attributes:
        use_mlflow: Whether to use MLflow for experiment tracking.
        use_wandb: Whether to use Weights & Biases for experiment tracking.
        wandb_project_name: Project name for Weights & Biases.
        mlflow_experiment_name: Experiment name for MLflow.
        mlflow_ignore_artifact_dirs: Directories to exclude from MLflow artifacts.
        log_params: List of custom parameters to log.
    """

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
    """Main configuration class for training experiments.

    This class contains all configuration parameters needed for training,
    validation, and testing of machine learning models.

    Attributes:
        epoch: Total number of training epochs.
        last_epoch: Last completed epoch (for resuming training).
        use_iter_loop: Whether to use iteration-based training instead of epoch-based.
        step_iter: Number of iterations per step.
        max_iter: Maximum number of iterations for iteration-based training.
        val_interval: Interval for running validation.
        save_interval: Interval for saving checkpoints.
        metric_for_best_model: Metric name to use for selecting best model.
        greater_is_better: Whether higher metric values are better.
        best_metric: Best metric value achieved so far.
        best_epoch: Epoch where best metric was achieved.
        batch: Batch size for training.
        num_worker: Number of data loader workers.
        use_ram_cache: Whether to use RAM caching for datasets.
        ram_cache_size_gb: Size of RAM cache in GB.
        seed: Random seed for reproducibility.
        gpu: GPU and distributed training configuration.
        use_amp: Whether to use Automatic Mixed Precision.
        amp_dtype: Data type for AMP (fp16 or bf16).
        amp_init_scale: Initial scale for AMP gradient scaler.
        use_compile: Whether to use torch.compile for model optimization.
        compile_backend: Backend for torch.compile.
        use_clip_grad: Whether to clip gradients.
        clip_grad_norm: Maximum gradient norm for clipping.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        output: Output directory for results.
        tag: Optional tag for the experiment.
        model: Model configuration.
        dataset: Dataset configuration.
        optimizer: Optimizer configuration.
        lr_scheduler: Learning rate scheduler configuration.
        evaluator: Evaluator configuration.
        log: Logging configuration.
    """

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
