import os
import logging

from dotenv import load_dotenv
import mlflow
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
import matplotlib.font_manager as font_manager  # noqa

# Get root logger
logger = logging.getLogger()


class TrainLogger:
    def __init__(self, cfg, output: str, tb_output: str) -> None:
        self.cfg = cfg
        self.output = output
        self.tb_output = tb_output
        os.makedirs(self.tb_output, exist_ok=True)

        self.tb_logger = TensorboardLogger(self.tb_output)

        self.mlflow_logger = None
        if self.cfg.USE_MLFLOW:
            experiment_name = self.cfg.EXPERIMENT_NAME
            run_name = os.path.basename(self.output)
            # run_name = os.path.basename(self.output).split("_")
            # run_name = "-".join(run_name[0:2])
            description = f"{self.cfg.MODEL.NAME} {self.cfg.DATASET.NAME} {self.cfg.TAG}"
            self.mlflow_logger = MlflowLogger(self.cfg, experiment_name, run_name, description)
            self.mlflow_logger.log_tag("phase", "Train")

        self.histories = {}
        self.last_epoch = 0

    def log_tag(self, key, value):
        if self.mlflow_logger:
            self.mlflow_logger.log_tag(key, value)

    def log_metric(self, name, metric, phase, step):
        self.tb_logger.write_scalars(name, {phase: metric}, step)
        if self.mlflow_logger:
            self.mlflow_logger.log_metric(f"{name}_{phase}", metric, step)

    def log_metrics(self, phase, metric_names, metric_values, step: int):
        self.last_epoch = step

        for name, value in zip(metric_names, metric_values):
            self.tb_logger.write_scalars(name, {phase: value}, step)
            if self.mlflow_logger:
                self.mlflow_logger.log_metric(f"{name}_{phase}", value, step)

            # Log value history
            if name not in self.histories:
                self.histories[name] = {}
            if phase not in self.histories[name]:
                self.histories[name][phase] = []
            self.histories[name][phase].append(value)

    def log_figure(self, tag, fig, path):
        self.tb_logger.write_figure(tag, fig)
        if self.mlflow_logger:
            self.mlflow_logger.log_figure(fig, path)

    def log_artifact(self, path):
        if self.mlflow_logger:
            self.mlflow_logger.log_artifact(path)

    def log_artifacts(self, path):
        if self.mlflow_logger:
            self.mlflow_logger.log_artifacts(path)

    def log_result_dir(self, path):
        """重みファイル(models以下)以外をartifactにする

        Args:
            path (str): result path
        """
        if not self.mlflow_logger:
            return
        for p in os.listdir(path):
            if os.path.isdir(p):
                if "models" in p:
                    continue
                else:
                    self.mlflow_logger.log_artifacts(p)
            else:
                self.mlflow_logger.log_artifact(p)

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
            labels = [f"{metric}_train", f"{metric}_val"]
            data = [self.histories[metric]["train"], self.histories[metric]["val"]]
            fig = self.plot_graph(metric, labels, data)
            fig_path = os.path.join(self.output, f"{metric}.png".replace(" ", "_"))
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
        self.tb_logger.writer_close()
        if self.mlflow_logger:
            self.mlflow_logger.close(status=status)


class TestLogger(TrainLogger):
    def __init__(self, cfg, output: str, tb_output: str) -> None:
        self.cfg = cfg
        self.output = output
        self.tb_output = tb_output
        os.makedirs(self.tb_output, exist_ok=True)

        self.tb_logger = TensorboardLogger(self.tb_output)

        self.mlflow_logger = None
        if self.cfg.USE_MLFLOW:
            experiment_name = os.path.basename(os.getcwd())
            run_name = os.path.basename(self.output).split("_")
            run_name = "_".join(run_name[0:2] + [self.cfg.MODEL.NAME, self.cfg.DATASET.NAME])
            description = f"{self.cfg.MODEL.NAME} {self.cfg.DATASET.NAME} {self.cfg.TAG} / {self.cfg.MODEL.WEIGHT}"
            self.mlflow_logger = MlflowLogger(self.cfg, experiment_name, run_name, description)
            self.mlflow_logger.log_tag("phase", "Test")
            self.mlflow_logger.log_tag("test weight path", self.cfg.MODEL.WEIGHT)

    def log_metrics(self, phase, metric_names, metric_values, step: int):
        for name, value in zip(metric_names, metric_values):
            self.tb_logger.write_scalars(name, {phase: value}, step)
            if self.mlflow_logger:
                self.mlflow_logger.log_metric(f"{name}_{phase}", value, step)


class TensorboardLogger:
    def __init__(self, output_dir):
        self.output_dir = os.path.join(output_dir, "tensorboard")
        self.writer = SummaryWriter(self.output_dir)

    def write_model_graph(self, model, device, input_size, multi_gpu=False):
        dummy = torch.zeros(input_size, device=device)
        model.eval()
        if multi_gpu:
            self.writer.add_graph(model.module, input_to_model=[dummy])
        else:
            self.writer.add_graph(model, input_to_model=[dummy])

    def write_scalars(self, tag, value_dict, step):
        self.writer.add_scalars(tag, value_dict, step)

    def write_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def write_figure(self, tag, fig):
        self.writer.add_figure(tag, fig)

    def writer_close(self):
        self.writer.close()


class MlflowLogger:
    def __init__(self, cfg, experiment_name, run_name, description) -> None:
        self.cfg = cfg
        load_dotenv(dotenv_path=f"{os.environ['HOME']}/.env")
        load_dotenv()

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not mlflow_uri:
            mlflow_uri = "./result/mlruns"
        mlflow.set_tracking_uri(mlflow_uri)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            # 当該Experiment存在しないとき、新たに作成
            self.experiment_id = mlflow.create_experiment(name=experiment_name)
        else:
            # 当該Experiment存在するとき、IDを取得
            self.experiment_id = self.experiment.experiment_id

        self.run = mlflow.start_run(
            experiment_id=self.experiment_id, run_name=run_name, description=description
        )

        logger.info(
            f"Start MLflow Tracking: experiment_name={experiment_name} run_name={run_name} experiment_id: {self.experiment_id} run_id: {self.run.info.run_id}"
        )

        self.log_hydra_config()

    def log_hydra_config(self):
        parameters = {
            "Optimizer": self.cfg.OPTIMIZER.NAME,
            "LR scheduler": self.cfg.LR_SCHEDULER,
            "Learning Rate": self.cfg.OPTIMIZER.LR,
            "Epoch": self.cfg.EPOCH,
            "Model": self.cfg.MODEL.NAME,
            "Dataset": self.cfg.DATASET.NAME,
            "Loss": self.cfg.LOSS.NAME,
            "Input size": self.cfg.MODEL.INPUT_SIZE,
            "PRE_TRAINED_WEIGHT": self.cfg.MODEL.PRE_TRAINED_WEIGHT,
            "Weight": self.cfg.MODEL.WEIGHT,
            "Batch": self.cfg.BATCH,
            "GPU Ids": self.cfg.GPU.USE,
            "hostname": os.uname()[1],
        }
        mlflow.log_params(parameters)

    def log_tag(self, key, value):
        mlflow.set_tag(key, value)

    def log_metric(self, name, value, step=None):
        mlflow.log_metric(name, value, step)

    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step)

    def log_param(self, name, value):
        mlflow.log_param(name, value)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_artifact(self, path):
        mlflow.log_artifact(path)

    def log_artifacts(self, path):
        mlflow.log_artifacts(path)

    def log_dict(self, data, path):
        mlflow.log_dict(data, path)

    def log_figure(self, fig, artifact_path):
        mlflow.log_figure(fig, artifact_path)

    def close(self, status):
        mlflow.end_run(status=status)
