import json
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from pprint import pformat
from typing import Literal, Optional, Union

import matplotlib
import matplotlib.figure
import matplotlib.font_manager as font_manager  # noqa
import matplotlib.pyplot as plt  # noqa
import mlflow
import numpy as np
import wandb
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import Tensor
from wandb.sdk.wandb_run import Run as WandbRun

from ..config import ExperimentConfig, LogParamsConfig
from ..types import PathLike, PhaseStr
from .torch_utils import is_main_process
from .utils import get_cmd, get_git_hash

matplotlib.use("Agg")
# 論文用にFontを変更する
font_manager.fontManager.addfont("./etc/Times_New_Roman.ttf")
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 18,
        # "text.usetex": True,
        "ps.useafm": True,
        "pdf.use14corefonts": True,
    }
)

load_dotenv(dotenv_path=f"{os.environ['HOME']}/.env")
load_dotenv()


class Logger:
    def __init__(
        self,
        output_dir: Optional[PathLike],
        phase: PhaseStr,
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        use_mlflow: bool = False,
        use_wandb: bool = False,
        mlflow_experiment_name: Optional[str] = None,
        wandb_project_name: Optional[str] = None,
    ) -> None:
        self.output_dir = output_dir
        self.histories = defaultdict(dict)
        self.last_epoch = 0
        self.phase = phase
        self.level = level
        self.use_mlflow = False
        self.use_wandb = False
        self.mlflow_run: Optional[mlflow.ActiveRun] = None
        self.wandb_run: Optional[WandbRun] = None

        log_path = output_dir / f"{self.phase}.log" if output_dir is not None else None
        self.logger = self.setup_logger(log_path, level)

        if is_main_process():
            with open(Path(output_dir) / "cmd_histry.log", "a") as f:
                print(get_cmd(), file=f)  # Execute CLI command history

            if use_mlflow and mlflow_experiment_name is not None and output_dir is not None:
                self.use_mlflow = True
                self.mlflow_run = self.setup_mlflow(output_dir.name, mlflow_experiment_name)
                self.logger.info(f"MLflow Tracking: {self.get_mlflow_run_uri()}")

            if use_wandb and wandb_project_name is not None and output_dir is not None:
                self.use_wandb = True
                self.wandb_run = self.setup_wandb(wandb_project_name, output_dir.name)
                self.logger.info(f"Wandb Tracking: {wandb.run.get_url()}")

        self.logger.info(f"Command: {get_cmd()}")
        self.logger.info(f"Git Hash: {get_git_hash()}")
        self.logger.info(f"Output dir: {str(output_dir)}")

    def setup_logger(self, log_path: Optional[Union[str, Path]] = None, level: str = "INFO"):
        logger.remove()
        logger.add(sys.stdout, level=level.upper())
        if log_path and is_main_process():
            logger.add(log_path, level=level.upper())
        return logger

    # ########## MLflow ##########
    def setup_mlflow(self, run_name: str, experiment_name: str):
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "./result/mlruns")
        mlflow.set_tracking_uri(mlflow_uri)

        experiment = mlflow.set_experiment(experiment_name=experiment_name)
        mlflow_run = mlflow.start_run(
            experiment_id=experiment.experiment_id, run_name=run_name, description=""
        )

        self.logger.info(
            f"Start MLflow Tracking: experiment_name={experiment_name} "
            f"run_name={run_name} "
            f"experiment_id: {experiment.experiment_id} run_id: {mlflow_run.info.run_id}"
        )
        self.log_params({"output_dir": self.output_dir, "hostname": os.uname()[1]})
        return mlflow_run

    def get_mlflow_run_uri(self) -> str:
        if not self.use_mlflow:
            return ""
        run = mlflow.get_run(self.mlflow_run.info.run_id)
        artifact_uri = run.info.artifact_uri

        if artifact_uri.startswith("file:"):
            return artifact_uri.replace("file:", "")
        else:
            tracking_uri = mlflow.get_tracking_uri()
            return f"{tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"

    @contextmanager
    def mlflow_safe_operation(self, operation_name: str):
        try:
            yield
        except Exception as e:
            self.logger.warning(f"Failed to {operation_name} on MLflow: {e}")

    # ########## Wandb ##########
    def setup_wandb(self, project_name: str, run_name: str) -> WandbRun:
        Path("result/wandb").mkdir(parents=True, exist_ok=True)
        wandb.require("core")
        return wandb.init(
            project=project_name,
            name=run_name,
            dir="result",
        )

    def get_wandb_run_uri(self) -> str:
        if self.use_wandb:
            return wandb.run.get_url()
        else:
            return ""

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def debug(self, message):
        self.logger.debug(message)

    def log_metric(
        self,
        name: str,
        metric: Union[int, float, Tensor],
        step: int,
        phase: Optional[PhaseStr] = None,
    ):
        if phase is None:
            phase = self.phase

        if isinstance(metric, Tensor):
            metric = metric.cpu().item()

        self.logger.info(f"{phase.capitalize()} {name}: {metric}")
        if self.use_mlflow:
            with self.mlflow_safe_operation("log_metric"):
                mlflow.log_metric(f"{phase}/{name}", metric, step)
        if self.use_wandb:
            wandb.log({f"{phase}/{name}": metric}, step=step)

    def log_metrics(self, metrics: dict, step: int, phase: Optional[PhaseStr] = None):
        if phase is None:
            phase = self.phase

        log_metrics = {}
        for name, value in metrics.items():
            if isinstance(value, Tensor):
                value = value.cpu().item()

            if isinstance(value, (int, float)):
                # Log value history
                if phase not in self.histories[name]:
                    self.histories[name][phase] = []
                self.histories[name][phase].append(value)

                log_metrics[f"{phase}/{name}"] = value

            self.logger.info(f"{phase.capitalize()} {name}: {value}")

        if self.use_mlflow:
            with self.mlflow_safe_operation("log_metrics"):
                mlflow.log_metrics(log_metrics, step)
        if self.use_wandb:
            wandb.log(log_metrics, step=step)

    def log_params(self, parameters: dict):
        if self.use_mlflow:
            with self.mlflow_safe_operation("log_params"):
                mlflow.log_params(parameters)
        if self.use_wandb:
            wandb.config.update(parameters)

    def log_tag(self, key: str, value: str):
        if self.use_mlflow:
            with self.mlflow_safe_operation("log_tag"):
                mlflow.set_tag(key, value)
        if self.use_wandb:
            wandb.run.tags[key] = value

    def log_figure(self, fig: matplotlib.figure.Figure, path: Union[str, Path]):
        if self.use_mlflow:
            with self.mlflow_safe_operation("log_figure"):
                mlflow.log_figure(fig, str(path))

    def log_artifact(self, path: Union[str, Path]):
        if self.use_mlflow:
            with self.mlflow_safe_operation("log_artifact"):
                mlflow.log_artifact(str(path))

    def log_artifacts(self, path: Union[str, Path]):
        if self.use_mlflow:
            with self.mlflow_safe_operation("log_artifacts"):
                mlflow.log_artifacts(str(path))

    def log_result_dir(self, path: Union[str, Path], ignore_dirs: list[str] = ["models"]):
        """重みファイル(models以下)以外をartifactにする

        Args:
            path (str): result path
        """
        if not self.use_mlflow:
            return

        for p in os.listdir(path):
            target = os.path.join(path, p)
            if os.path.isdir(target):
                ignore = False
                for ignore_dir_name in ignore_dirs:
                    if ignore_dir_name in target:
                        ignore = True
                        continue
                if ignore:
                    continue
                with self.mlflow_safe_operation("log_artifacts"):
                    mlflow.log_artifacts(target)
            else:
                with self.mlflow_safe_operation("log_artifact"):
                    mlflow.log_artifact(target)

    def log_table(self, dict_data):
        if self.use_mlflow:
            with self.mlflow_safe_operation("log_table"):
                mlflow.log_table(data=dict_data)

    def log_history_figure(self):
        self.logger.info(f"Histories:\n{pformat(self.histories)}")
        metrics_names = list(self.histories.keys())

        for metric in metrics_names:
            labels = []
            data = []
            for k, v in self.histories[metric].items():
                labels.append(f"{metric}_{k}")
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

            fig = self.plot_graph(metric, labels, interpolated_data)
            fig_path = os.path.join(self.output_dir, f"{metric}.png".replace(" ", "_"))
            fig.savefig(fig_path)
            plt.close()

    def plot_graph(self, title, labels, data):
        plt.gcf().clear()

        fig, ax = plt.subplots(
            1, 1, figsize=(9, 6), tight_layout=True, subplot_kw=dict(title=title)
        )
        for label, d in zip(labels, data):
            ax.plot(range(len(d)), d, label=label)
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, len(data[0]) - 1)  # x軸の範囲を設定
        return fig

    def log_config(self, cfg: ExperimentConfig, params: LogParamsConfig):
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

    def close(self, status="FINISHED"):
        if self.use_mlflow:
            mlflow.end_run(status=status)
        if self.use_wandb:
            wandb.finish(exit_code=0 if status == "FINISHED" else 1)
