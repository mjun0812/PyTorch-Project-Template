import logging
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
from dotenv import load_dotenv
from torch import Tensor

from .utils import get_cmd, get_git_hash

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa
import matplotlib.pyplot as plt  # noqa

load_dotenv(dotenv_path=f"{os.environ['HOME']}/.env")
load_dotenv()


class Logger:
    def __init__(self, output_dir: str, log_path: str, phase: str, level="INFO") -> None:
        self.output_dir = output_dir
        self.histories = defaultdict(dict)
        self.last_epoch = 0
        self.use_mlflow = False
        self.phase = phase
        self.level = level  # INFO, DEBUG, WARNING, ERROR, CRITICAL

        self.logger = self.setup_logger(log_path, level)

        self.logger.info(f"Command: {get_cmd()}")
        self.logger.info(f"Git Hash: {get_git_hash()}")
        self.logger.info(f"Output dir: {str(output_dir)}")

    def setup_logger(self, log_path: str = None, level="INFO"):
        logger = logging.getLogger()
        logger.setLevel(level.upper())
        log_format = "[%(asctime)s][%(levelname)s] %(message)s"

        # 既存のハンドラを削除
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])

        # コンソール出力用のハンドラを設定
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        console_handler.setLevel(level.upper())
        logger.addHandler(console_handler)

        if log_path:
            file_handler = logging.FileHandler(log_path, "a")
            file_handler.setFormatter(logging.Formatter(log_format))
            file_handler.setLevel(level.upper())
            logger.addHandler(file_handler)
        return logger

    def setup_mlflow(self, run_name, experiment_name):
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "./result/mlruns")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.enable_system_metrics_logging()

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # 当該Experimentが存在しないとき、新たに作成
            experiment_id = mlflow.create_experiment(name=experiment_name)
        else:
            # 当該Experimentが存在するとき、IDを取得
            experiment_id = experiment.experiment_id

        mlflow_run = mlflow.start_run(
            experiment_id=experiment_id, run_name=run_name, description=""
        )

        self.use_mlflow = True
        self.logger.info(
            f"Start MLflow Tracking: experiment_name={experiment_name} "
            f"run_name={run_name} experiment_id: {experiment_id} run_id: {mlflow_run.info.run_id}"
        )
        self.log_params({"output_dir": self.output_dir})
        return mlflow_run

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

    def log_metric(self, name, metric, step):
        if isinstance(metric, Tensor):
            metric = metric.cpu().item()

        self.logger.info(f"{self.phase.capitalize()} {name}: {metric}")
        if self.use_mlflow:
            mlflow.log_metric(f"{name}_{self.phase}", metric, step)

    def log_metrics(self, metrics: dict, step: int):
        mlflow_metrics = {}
        for name, value in metrics.items():
            if isinstance(value, Tensor):
                value = value.cpu().item()

            self.logger.info(f"{self.phase.capitalize()} {name}: {value}")

            if isinstance(value, (int, float)):
                # Log value history
                if self.phase not in self.histories[name]:
                    self.histories[name][self.phase] = []
                self.histories[name][self.phase].append(value)

                mlflow_metrics[f"{name}_{self.phase}"] = value

        if self.use_mlflow:
            mlflow.log_metrics(mlflow_metrics, step)

    def log_params(self, parameters: dict):
        if self.use_mlflow:
            mlflow.log_params(parameters)

    def log_tag(self, key, value):
        if self.use_mlflow:
            mlflow.set_tag(key, value)

    def log_figure(self, fig, path):
        if self.use_mlflow:
            mlflow.log_figure(fig, path)

    def log_artifact(self, path):
        if isinstance(path, Path):
            path = str(path)
        if self.use_mlflow:
            mlflow.log_artifact(path)

    def log_artifacts(self, path):
        if isinstance(path, Path):
            path = str(path)
        if self.use_mlflow:
            mlflow.log_artifacts(path)

    def log_result_dir(self, path, ignore_dirs=["models"]):
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

    def close(self, status="FINISHED"):
        if self.use_mlflow:
            mlflow.end_run(status=status)
