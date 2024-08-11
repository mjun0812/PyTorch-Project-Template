import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib
import matplotlib.figure
import mlflow
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from torch import Tensor

from ..alias import PhaseStr
from ..config import ExperimentConfig, MlflowLogParamsConfig
from .torch_utils import is_main_process
from .utils import get_cmd, get_git_hash

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa
import matplotlib.pyplot as plt  # noqa

load_dotenv(dotenv_path=f"{os.environ['HOME']}/.env")
load_dotenv()


class Logger:
    def __init__(
        self,
        output_dir: Union[str, Path],
        phase: PhaseStr,
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        use_mlflow: bool = False,
        mlflow_experiment_name: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.histories = defaultdict(dict)
        self.last_epoch = 0
        self.phase = phase
        self.level = level
        self.use_mlflow = False

        self.logger = self.setup_logger(output_dir / f"{self.phase}.log", level)

        if is_main_process():
            with open(Path(output_dir) / "cmd_histry.log", "a") as f:
                print(get_cmd(), file=f)  # Execute CLI command history

        if use_mlflow and is_main_process():
            self.use_mlflow = True
            self.mlflow_run = self.setup_mlflow(output_dir.name, mlflow_experiment_name)
            self.logger.info(f"MLflow Tracking URI: {self.get_mlflow_run_uri()}")

        self.logger.info(f"Command: {get_cmd()}")
        self.logger.info(f"Git Hash: {get_git_hash()}")
        self.logger.info(f"Output dir: {str(output_dir)}")

    def setup_logger(self, log_path: Optional[Union[str, Path]] = None, level: str = "INFO"):
        logger.remove()
        logger.add(sys.stdout, level=level.upper())
        if log_path and is_main_process():
            logger.add(log_path, level=level.upper())
        return logger

    def setup_mlflow(self, run_name: str, experiment_name: str):
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "./result/mlruns")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.enable_system_metrics_logging()

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
        run = mlflow.get_run(self.mlflow_run.info.run_id)
        artifact_uri = run.info.artifact_uri

        if artifact_uri.startswith("file:"):
            return artifact_uri.replace("file:", "")
        else:
            tracking_uri = mlflow.get_tracking_uri()
            return f"{tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"

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
            mlflow.log_metric(f"{name}_{phase}", metric, step)

    def log_metrics(self, metrics: dict, step: int, phase: Optional[PhaseStr] = None):
        if phase is None:
            phase = self.phase

        mlflow_metrics = {}
        for name, value in metrics.items():
            if isinstance(value, Tensor):
                value = value.cpu().item()

            if isinstance(value, (int, float)):
                # Log value history
                if phase not in self.histories[name]:
                    self.histories[name][phase] = []
                self.histories[name][phase].append(value)

                mlflow_metrics[f"{name}_{phase}"] = value

            self.logger.info(f"{phase.capitalize()} {name}: {value}")

        if self.use_mlflow:
            mlflow.log_metrics(mlflow_metrics, step)

    def log_params(self, parameters: dict):
        if self.use_mlflow:
            mlflow.log_params(parameters)

    def log_tag(self, key: str, value: str):
        if self.use_mlflow:
            mlflow.set_tag(key, value)

    def log_figure(self, fig: matplotlib.figure.Figure, path: Union[str, Path]):
        if self.use_mlflow:
            mlflow.log_figure(fig, str(path))

    def log_artifact(self, path: Union[str, Path]):
        if self.use_mlflow:
            mlflow.log_artifact(str(path))

    def log_artifacts(self, path: Union[str, Path]):
        if self.use_mlflow:
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
                mlflow.log_artifacts(target)
            else:
                mlflow.log_artifact(target)

    def log_table(self, dict_data):
        if self.use_mlflow:
            mlflow.log_table(data=dict_data)

    def log_history_figure(self):
        metrics_names = list(self.histories.keys())
        # サンプル不足でのグラフ描画エラーの処理
        # Validation Intervalによってはlen(hist_loss) > len(hist_val_loss)
        # なので、x軸を補間することで、グラフを合わせる
        x = np.arange(self.last_epoch)
        val_x = np.linspace(0, self.last_epoch, len(self.histories[metrics_names[0]]["val"]))
        for name in metrics_names:
            self.histories[name]["val"] = np.interp(x, val_x, self.histories[name]["val"])

        for metric in metrics_names:
            labels = []
            data = []
            for k, v in self.histories[metric].items():
                labels.append(f"{metric}_{k}")
                data.append(v)
            fig = self.plot_graph(metric, labels, data)
            fig_path = os.path.join(self.output_dir, f"{metric}.png".replace(" ", "_"))
            fig.savefig(fig_path)
            plt.close()

    def plot_graph(self, title, labels, data):
        plt.gcf().clear()

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
        fig, ax = plt.subplots(
            1, 1, figsize=(9, 6), tight_layout=True, subplot_kw=dict(title=title)
        )
        for label, d in zip(labels, data):
            ax.plot(d, label=label)
        ax.legend()
        return fig

    def log_config(self, cfg: ExperimentConfig, params: MlflowLogParamsConfig):
        log_params = {}
        for p in params:
            log_params[p.name] = OmegaConf.select(cfg, p.value)
        self.log_params(log_params)

    def close(self, status="FINISHED"):
        if self.use_mlflow:
            mlflow.end_run(status=status)
