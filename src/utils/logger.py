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
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import Tensor

from ..config import ExperimentConfig, LogParamsConfig
from ..types import PathLike, PhaseStr
from .utils import get_cmd, get_git_hash

matplotlib.use("Agg")
# 論文用にFontを変更する
font_manager.fontManager.addfont("./etc/Times_New_Roman.ttf")
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 18})

load_dotenv()


class MlflowLogger:
    def __init__(self, experiment_name: str, run_name: str) -> None:
        self.run = self.setup(experiment_name, run_name)

    def setup(self, experiment_name: str, run_name: str) -> mlflow.ActiveRun:
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "./result/mlruns")
        mlflow.set_tracking_uri(mlflow_uri)
        experiment = mlflow.set_experiment(experiment_name=experiment_name)
        return mlflow.start_run(
            experiment_id=experiment.experiment_id, run_name=run_name, description=""
        )

    def get_run_uri(self) -> str:
        run = mlflow.get_run(self.run.info.run_id)
        artifact_uri = run.info.artifact_uri

        if artifact_uri.startswith("file:"):
            return artifact_uri.replace("file:", "")
        else:
            tracking_uri = mlflow.get_tracking_uri()
            return f"{tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"

    def log_metric(self, name: str, metric: int | float | Tensor, step: int) -> None:
        mlflow.log_metric(name, metric, step)

    def log_metrics(self, metrics: dict, step: int) -> None:
        mlflow.log_metrics(metrics, step)

    def log_params(self, parameters: dict) -> None:
        mlflow.log_params(parameters)

    def log_tag(self, key: str, value: str) -> None:
        mlflow.set_tag(key, value)

    def log_artifact(self, path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    def log_figure(self, fig: matplotlib.figure.Figure, path: str | Path) -> None:
        mlflow.log_figure(fig, str(path))

    def log_artifacts(self, path: str | Path) -> None:
        mlflow.log_artifacts(str(path))

    def log_table(self, dict_data: dict[str, Any]) -> None:
        mlflow.log_table(data=dict_data)

    def close(self, status: Literal["FINISHED", "FAILED"] = "FINISHED") -> None:
        mlflow.end_run(status=status)


class WandbLogger:
    def __init__(self, project_name: str, run_name: str) -> None:
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
    return {k: defaultdict_to_dict(v) if isinstance(v, defaultdict) else v for k, v in d.items()}


class MetricLogger:
    def __init__(self) -> None:
        self.histories = defaultdict(lambda: defaultdict(list))

    def log_metric(self, metric_name: str, metric: int | float | str, phase: PhaseStr) -> None:
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
    def __init__(
        self,
        output_dir: PathLike | None,
        phase: PhaseStr,
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        use_mlflow: bool = False,
        use_wandb: bool = False,
        mlflow_experiment_name: str | None = None,
        wandb_project_name: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.last_epoch = 0
        self.phase = phase
        self.level = level

        # 基本ロガーの設定
        self._setup_basic_logger(output_dir, level)

        # メトリックロガーの初期化
        self.metric_logger = MetricLogger()

        # 外部ロガーの設定
        self._setup_mlflow_logger(use_mlflow, mlflow_experiment_name, output_dir)
        self._setup_wandb_logger(use_wandb, wandb_project_name, output_dir)

        self._log_initial_info()

    def _setup_basic_logger(self, output_dir: PathLike | None, level: str) -> None:
        """基本ロガーを設定する"""
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
        self, use_mlflow: bool, experiment_name: str | None, output_dir: PathLike | None
    ) -> None:
        """MLflowロガーを設定する"""
        self.mlflow_logger: MlflowLogger | None = None
        if use_mlflow and experiment_name is not None and output_dir is not None:
            self.mlflow_logger = MlflowLogger(experiment_name, output_dir.name)
            self.logger.info(f"MLflow Tracking: {self.mlflow_logger.get_run_uri()}")
            self.logger.info(
                f"Start MLflow Tracking: experiment_name={experiment_name} "
                f"run_name={output_dir.name} "
                f"experiment_id: {self.mlflow_logger.run.info.experiment_id} "
                f"run_id: {self.mlflow_logger.run.info.run_id}"
            )
            self.mlflow_logger.log_params(
                {"output_dir": self.output_dir, "hostname": os.uname()[1]}
            )

    def _setup_wandb_logger(
        self, use_wandb: bool, project_name: str | None, output_dir: PathLike | None
    ) -> None:
        """Weights & Biasesロガーを設定する"""
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
        """結果ディレクトリをログに記録する

        Args:
            path (Union[str, Path]): 結果ディレクトリのパス
            ignore_dirs (Optional[list[str]], optional): 無視するディレクトリ名のリスト
        """
        if self.mlflow_logger is None:
            return

        if ignore_dirs is None:
            ignore_dirs = []

        path = Path(path)

        for p in path.iterdir():
            target = path / p

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

    def close(self, status: str = "FINISHED") -> None:
        if self.mlflow_logger:
            with self._safe_operation("mlflow.end_run"):
                self.mlflow_logger.close(status)
        if self.wandb_logger:
            with self._safe_operation("wandb.finish"):
                self.wandb_logger.close(status)
