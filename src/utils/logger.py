import json
import os
import sys
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from pprint import pformat
from typing import Any, Literal

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import mlflow
import mlflow.runs
import numpy as np
import wandb
from dotenv import load_dotenv
from loguru import logger
from matplotlib import font_manager
from mlflow.data.dataset_source_registry import resolve_dataset_source
from mlflow.data.meta_dataset import MetaDataset
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import Tensor
from torch.nn import Module as TorchModule

from ..config import ExperimentConfig, LogParamsConfig
from ..dataloaders.base import BaseDataset
from ..dataloaders.types import DatasetOutput
from ..types import PathLike, PhaseStr
from .utils import get_cmd, get_git_hash

load_dotenv()


class MlflowLogger:
    """MLflow logger for experiment tracking.

    Provides methods to log metrics, parameters, artifacts, and figures
    to MLflow tracking server.

    Attributes:
        run: Active MLflow run instance.
        model_info: Model information.
    """

    def __init__(self, experiment_name: str, run_name: str) -> None:
        """Initialize MLflow logger.

        Args:
            experiment_name: Name of the MLflow experiment.
            run_name: Name of the MLflow run.
        """
        self.run = self.setup(experiment_name, run_name)
        self.model_info = None

    def setup(self, experiment_name: str, run_name: str) -> mlflow.ActiveRun:
        """Set up MLflow tracking.

        Args:
            experiment_name: Name of the MLflow experiment.
            run_name: Name of the MLflow run.

        Returns:
            Active MLflow run instance.
        """
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "./result/mlruns")
        mlflow.set_tracking_uri(mlflow_uri)
        experiment = mlflow.set_experiment(experiment_name=experiment_name)
        return mlflow.start_run(
            experiment_id=experiment.experiment_id, run_name=run_name, description=""
        )

    def get_run_uri(self) -> str:
        """Get the URI of the MLflow run.

        Returns:
            str: The URI of the MLflow run.
        """
        run = mlflow.get_run(self.run.info.run_id)
        artifact_uri = run.info.artifact_uri

        if artifact_uri.startswith("file:"):
            return artifact_uri.replace("file:", "")
        else:
            tracking_uri = mlflow.get_tracking_uri()
            return f"{tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"

    def log_metric(self, name: str, metric: int | float | Tensor, step: int) -> None:
        """Log a single metric value.

        Args:
            name: Name of the metric.
            metric: Value of the metric.
            step: Step number.
        """
        mlflow.log_metric(name, metric, step)

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log multiple metric values.

        Args:
            metrics: Dictionary of metric names and values.
            step: Step number.
        """
        mlflow.log_metrics(metrics, step)

    def log_params(self, parameters: dict) -> None:
        """Log parameters.

        Args:
            parameters: Dictionary of parameter names and values.
        """
        mlflow.log_params(parameters)

    def log_tag(self, key: str, value: str) -> None:
        """Log a tag.

        Args:
            key: Name of the tag.
            value: Value of the tag.
        """
        mlflow.set_tag(key, value)

    def log_artifact(self, path: str | Path) -> None:
        """Log an artifact.

        Args:
            path: Path to the artifact.
        """
        mlflow.log_artifact(str(path))

    def log_figure(self, fig: matplotlib.figure.Figure, path: str | Path) -> None:
        """Log a figure.

        Args:
            fig: Figure to log.
            path: Path to the figure.
        """
        mlflow.log_figure(fig, str(path))

    def log_artifacts(self, path: str | Path) -> None:
        """Log multiple artifacts.

        Args:
            path: Path to the artifacts.
        """
        mlflow.log_artifacts(str(path))

    def log_table(self, dict_data: dict[str, Any]) -> None:
        mlflow.log_table(data=dict_data)

    def log_model(self, model: TorchModule, model_name: str) -> None:
        input_example = {key: None for key in DatasetOutput.__annotations__}
        self.model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            input_example=input_example,
            name=model_name,
            registered_model_name=model_name,
        )

    def log_dataset(self, dataset: BaseDataset, name: str, context: str | None) -> None:
        dataset_source = resolve_dataset_source(dataset)
        meta_dataset = MetaDataset(name=name, source=dataset_source)
        mlflow.log_input(dataset=meta_dataset, context=context, model=self.model_info)

    def close(self, status: Literal["FINISHED", "FAILED"] = "FINISHED") -> None:
        mlflow.end_run(status=status)


class WandbLogger:
    """Weights & Biases logger for experiment tracking.

    Provides methods to log metrics, parameters, and other experiment data
    to Weights & Biases.

    Attributes:
        run: Active Weights & Biases run instance.
    """

    def __init__(self, project_name: str, run_name: str) -> None:
        """Initialize Weights & Biases logger.

        Args:
            project_name: Name of the W&B project.
            run_name: Name of the W&B run.
        """
        self.run = self.setup(project_name, run_name)

    def setup(self, project_name: str, run_name: str) -> wandb.sdk.wandb_run.Run:
        Path("result/wandb").mkdir(parents=True, exist_ok=True)
        wandb.require("core")
        return wandb.init(project=project_name, name=run_name, dir="result")

    def get_run_uri(self) -> str:
        return self.run.get_url()

    def log_metric(self, name: str, metric: int | float | Tensor, step: int) -> None:
        wandb.log({name: metric}, step=step)

    def log_metrics(self, metrics: dict, step: int) -> None:
        wandb.log(metrics, step=step)

    def log_params(self, parameters: dict) -> None:
        wandb.config.update(parameters)

    def log_tag(self, key: str, value: str) -> None:
        wandb.run.tags.append(value)

    def close(self, status: Literal["FINISHED", "FAILED"] = "FINISHED") -> None:
        wandb.finish(exit_code=0 if status == "FINISHED" else 1)


def defaultdict_to_dict(d: defaultdict) -> dict:
    """Convert defaultdict to regular dict recursively.

    Args:
        d: The defaultdict to convert.

    Returns:
        Regular dictionary with all nested defaultdicts converted.
    """
    return {k: defaultdict_to_dict(v) if isinstance(v, defaultdict) else v for k, v in d.items()}


class MetricLogger:
    """Logger for tracking and visualizing training metrics.

    Stores metric histories for different phases (train, val, test) and
    provides functionality to generate plots and figures.

    Attributes:
        histories: Nested defaultdict storing metric values by name and phase.
    """

    def __init__(self) -> None:
        """Initialize the metric logger."""
        self.histories = defaultdict(lambda: defaultdict(list))

    def log_metric(self, metric_name: str, metric: int | float | str, phase: PhaseStr) -> None:
        """Log a single metric value.

        Args:
            metric_name: Name of the metric.
            metric: Value of the metric.
            phase: Training phase (train, val, test).
        """
        self.histories[metric_name][phase].append(metric)

    def log_metrics(self, metrics: dict[str, int | float | str], phase: PhaseStr) -> None:
        for name, value in metrics.items():
            self.log_metric(phase, name, value)

    def _plot_graph(
        self, title: str, labels: list[str], data: list[list[int | float]]
    ) -> matplotlib.figure.Figure:
        plt.gcf().clear()
        fig, ax = plt.subplots(
            1, 1, figsize=(9, 6), tight_layout=True, subplot_kw=dict(title=title)
        )
        for label, d in zip(labels, data):
            ax.plot(range(len(d)), d, label=label)
        ax.legend()
        ax.set_xlabel("Step")
        ax.set_xlim(0, len(data[0]))
        return fig

    def log_history_figure(self, output_dir: PathLike) -> None:
        matplotlib.use("Agg")
        # 論文用にFontを変更する
        font_manager.fontManager.addfont("./etc/Times_New_Roman.ttf")
        plt.rcParams.update({"font.family": "Times New Roman", "font.size": 18})

        metrics_names = list(self.histories.keys())
        output_dir = Path(output_dir)

        for metric_name in metrics_names:
            labels = []
            data = []
            for phase, v in self.histories[metric_name].items():
                labels.append(f"{metric_name}_{phase}")
                data.append(v)
            if not data:
                continue
            max_length = max([len(d) for d in data])

            # サンプル不足でのグラフ描画エラーの処理
            # Validation Intervalによってはlen(hist_loss) > len(hist_val_loss)
            # なので、x軸を補間することで、グラフを合わせる
            interpolated_data = []
            for d in data:
                if len(d) < max_length:
                    x = np.linspace(0, max_length - 1, len(d))
                    interp_data = np.interp(np.arange(max_length), x, d)
                    interpolated_data.append(interp_data)
                else:
                    interpolated_data.append(d)

            fig = self._plot_graph(metric_name, labels, interpolated_data)
            fig_path = output_dir / f"{metric_name}.png".replace(" ", "_")
            fig.savefig(fig_path)
            plt.close()


class Logger:
    """Unified logger that combines multiple logging backends.

    Integrates basic logging, metric tracking, MLflow, and Weights & Biases
    into a single interface for experiment logging.

    Attributes:
        output_dir: Directory for saving logs and artifacts.
        last_epoch: Last completed epoch number.
        phase: Current training phase.
        level: Logging level.
        metric_logger: Instance for tracking metric histories.
        mlflow_logger: MLflow logger instance if enabled.
        wandb_logger: Weights & Biases logger instance if enabled.
        logger: Basic logger instance.
        log_path: Path to the log file.
    """

    def __init__(
        self,
        output_dir: PathLike | None,
        phase: PhaseStr,
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        use_mlflow: bool = False,
        use_wandb: bool = False,
        mlflow_experiment_name: str | None = None,
        mlflow_run_name: str | None = None,
        wandb_project_name: str | None = None,
    ) -> None:
        """Initialize the unified logger.

        Args:
            output_dir: Directory for saving logs and artifacts.
            phase: Training phase (train, val, test).
            level: Logging level.
            use_mlflow: Whether to enable MLflow logging.
            use_wandb: Whether to enable Weights & Biases logging.
            mlflow_experiment_name: Name of the MLflow experiment.
            mlflow_run_name: Name of the MLflow run.
            wandb_project_name: Name of the W&B project.
        """
        self.output_dir = output_dir
        self.last_epoch = 0
        self.phase = phase
        self.level = level

        # 基本ロガーの設定
        self._setup_basic_logger(output_dir, level)

        # メトリックロガーの初期化
        self.metric_logger = MetricLogger()

        # 外部ロガーの設定
        if mlflow_run_name is None and output_dir is not None:
            mlflow_run_name = Path(output_dir).name
        self._setup_mlflow_logger(use_mlflow, mlflow_experiment_name, mlflow_run_name)
        self._setup_wandb_logger(use_wandb, wandb_project_name, output_dir)

        self._log_initial_info()

    def _setup_basic_logger(self, output_dir: PathLike | None, level: str) -> None:
        """Set up the basic logger.

        Args:
            output_dir: Directory for saving log files.
            level: Logging level.
        """
        if output_dir is not None:
            self.log_path = output_dir / f"{self.phase}.log"
        else:
            self.log_path = None

        logger.remove()
        logger.add(sys.stdout, level=level.upper())
        if self.log_path:
            logger.add(self.log_path, level=level.upper())
        self.logger = logger

    def _setup_mlflow_logger(
        self, use_mlflow: bool, experiment_name: str | None, run_name: str | None
    ) -> None:
        """Set up the MLflow logger.

        Args:
            use_mlflow: Whether to enable MLflow logging.
            experiment_name: Name of the MLflow experiment.
            run_name: Name of the MLflow run.
        """
        self.mlflow_logger: MlflowLogger | None = None
        if use_mlflow and experiment_name is not None and run_name is not None:
            self.mlflow_logger = MlflowLogger(experiment_name, run_name)
            self.logger.info(f"MLflow Tracking: {self.mlflow_logger.get_run_uri()}")
            self.logger.info(
                f"Start MLflow Tracking: experiment_name={experiment_name} "
                f"run_name={run_name} "
                f"experiment_id: {self.mlflow_logger.run.info.experiment_id} "
                f"run_id: {self.mlflow_logger.run.info.run_id}"
            )
            self.mlflow_logger.log_params(
                {"output_dir": self.output_dir, "hostname": os.uname()[1]}
            )

    def _setup_wandb_logger(
        self, use_wandb: bool, project_name: str | None, output_dir: PathLike | None
    ) -> None:
        """Set up the Weights & Biases logger.

        Args:
            use_wandb: Whether to enable W&B logging.
            project_name: Name of the W&B project.
            output_dir: Directory for saving artifacts.
        """
        self.wandb_logger: WandbLogger | None = None
        if use_wandb and project_name is not None and output_dir is not None:
            self.wandb_logger = WandbLogger(project_name, output_dir.name)
            self.logger.info(f"Wandb Tracking: {self.wandb_logger.get_run_uri()}")

    def _log_initial_info(self) -> None:
        if self.output_dir is not None:
            with open(Path(self.output_dir) / "cmd_histry.log", "a") as f:
                print(get_cmd(), file=f)  # Execute CLI command history
        self.logger.info(f"Command: {get_cmd()}")
        self.logger.info(f"Git Hash: {get_git_hash()}")
        self.logger.info(f"Output dir: {self.output_dir!s}")

    @contextmanager
    def _safe_operation(self, operation_name: str) -> Generator[None, None, None]:
        try:
            yield
        except Exception as e:
            self.logger.warning(f"Failed to {operation_name}: {e}")

    def get_mlflow_run_uri(self) -> str | None:
        if self.mlflow_logger is None:
            return None
        return self.mlflow_logger.get_run_uri()

    def get_wandb_run_uri(self) -> str | None:
        if self.wandb_logger is None:
            return None
        return self.wandb_logger.get_run_uri()

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def exception(self, message: str) -> None:
        self.logger.exception(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def log_metric(
        self,
        name: str,
        metric: int | float | Tensor,
        step: int,
        phase: PhaseStr | None = None,
    ) -> None:
        if phase is None:
            phase = self.phase
        if isinstance(metric, Tensor):
            metric = metric.cpu().item()

        self.logger.info(f"{phase.capitalize()} {name}: {metric}")
        self.metric_logger.log_metric(name, metric, phase)

        if self.mlflow_logger:
            with self._safe_operation("mlflow.log_metric"):
                self.mlflow_logger.log_metric(f"{phase}/{name}", metric, step)
        if self.wandb_logger:
            with self._safe_operation("wandb.log"):
                self.wandb_logger.log_metric(f"{phase}/{name}", metric, step)

    def log_metrics(self, metrics: dict, step: int, phase: PhaseStr | None = None) -> None:
        if phase is None:
            phase = self.phase

        log_metrics = {}
        for name, value in metrics.items():
            if isinstance(value, Tensor):
                value = value.cpu().item()

            if isinstance(value, (int, float, str)):
                log_metrics[f"{phase}/{name}"] = value

                if not isinstance(value, str):
                    self.metric_logger.log_metric(name, value, phase)

            self.logger.info(f"{phase.capitalize()} {name}: {value}")

        if self.mlflow_logger:
            with self._safe_operation("mlflow.log_metrics"):
                self.mlflow_logger.log_metrics(log_metrics, step)
        if self.wandb_logger:
            with self._safe_operation("wandb.log"):
                self.wandb_logger.log_metrics(log_metrics, step)

    def log_params(self, parameters: dict) -> None:
        if self.mlflow_logger:
            with self._safe_operation("mlflow.log_params"):
                self.mlflow_logger.log_params(parameters)
        if self.wandb_logger:
            with self._safe_operation("wandb.log_params"):
                self.wandb_logger.log_params(parameters)

    def log_tag(self, key: str, value: str) -> None:
        if self.mlflow_logger:
            with self._safe_operation("mlflow.log_tag"):
                self.mlflow_logger.log_tag(key, value)
        if self.wandb_logger:
            with self._safe_operation("wandb.log_tag"):
                self.wandb_logger.log_tag(key, value)

    def log_figure(self, fig: matplotlib.figure.Figure, path: str | Path) -> None:
        if self.mlflow_logger:
            with self._safe_operation("mlflow.log_figure"):
                self.mlflow_logger.log_figure(fig, str(path))

    def log_artifact(self, path: str | Path) -> None:
        if self.mlflow_logger:
            with self._safe_operation("mlflow.log_artifact"):
                self.mlflow_logger.log_artifact(str(path))

    def log_artifacts(self, path: str | Path) -> None:
        if self.mlflow_logger:
            with self._safe_operation("mlflow.log_artifacts"):
                self.mlflow_logger.log_artifacts(str(path))

    def log_result_dir(self, path: str | Path, ignore_dirs: list[str] | None = None) -> None:
        """Log all files and directories in the result directory.

        Args:
            path: Path to the result directory.
            ignore_dirs: List of directory names to ignore.
        """
        if self.mlflow_logger is None:
            return

        if ignore_dirs is None:
            ignore_dirs = []

        path = Path(path)

        for target in path.iterdir():
            if target.is_dir():
                # ディレクトリが無視リストに含まれているかチェック
                if any(ignore_dir_name in str(target) for ignore_dir_name in ignore_dirs):
                    continue

                with self._safe_operation("mlflow.log_artifacts"):
                    self.mlflow_logger.log_artifacts(target)
            else:
                with self._safe_operation("mlflow.log_artifact"):
                    self.mlflow_logger.log_artifact(target)

    def log_table(self, dict_data: dict[str, Any]) -> None:
        if self.mlflow_logger:
            with self._safe_operation("mlflow.log_table"):
                self.mlflow_logger.log_table(data=dict_data)

    def log_history_figure(self, output_dir: PathLike) -> None:
        histories = defaultdict_to_dict(self.metric_logger.histories)
        self.logger.info(f"Histories:\n{pformat(histories)}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with self._safe_operation("log_history_figure"):
            self.metric_logger.log_history_figure(output_dir)

    def log_config(self, cfg: ExperimentConfig, params: LogParamsConfig) -> None:
        log_params = {}
        for p in params:
            value = OmegaConf.select(cfg, p.value)
            if isinstance(value, (list, tuple, ListConfig)):
                log_params[p.name] = value
            elif isinstance(value, (DictConfig, dict)):
                log_params[p.name] = json.dumps(value, ensure_ascii=False)
            else:
                log_params[p.name] = value
        self.log_params(log_params)

    def log_model(self, model: TorchModule, model_name: str) -> None:
        if self.mlflow_logger:
            with self._safe_operation("mlflow.log_model"):
                self.mlflow_logger.log_model(model, model_name)

    def log_dataset(self, dataset: BaseDataset, name: str, context: str | None) -> None:
        if self.mlflow_logger:
            with self._safe_operation("mlflow.log_dataset"):
                self.mlflow_logger.log_dataset(dataset, name, context)

    def close(self, status: str = "FINISHED") -> None:
        if self.mlflow_logger:
            with self._safe_operation("mlflow.end_run"):
                self.mlflow_logger.close(status)
        if self.wandb_logger:
            with self._safe_operation("wandb.finish"):
                self.wandb_logger.close(status)
