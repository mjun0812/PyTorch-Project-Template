import logging
import os

import matplotlib
import mlflow
import numpy as np
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa
import matplotlib.pyplot as plt  # noqa

# Get root logger
logger = logging.getLogger()
load_dotenv(dotenv_path=f"{os.environ['HOME']}/.env")
load_dotenv()


class Writer:
    def __init__(self, cfg, output: str, phase: str) -> None:
        self.cfg = cfg
        self.output = output
        self.histories = {}
        self.last_epoch = 0
        self.use_mlflow = self.cfg.USE_MLFLOW
        self.phase = phase

        if self.use_mlflow:
            self.setup_mlflow()

    def setup_mlflow(self):
        run_name = os.path.basename(self.output).split("_")
        run_name = "_".join(run_name[0:2] + [self.cfg.MODEL.NAME, self.cfg.DATASET.NAME])

        experiment_name = self.cfg.EXPERIMENT_NAME

        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "./result/mlruns")
        mlflow.set_tracking_uri(mlflow_uri)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # 当該Experimentが存在しないとき、新たに作成
            experiment_id = mlflow.create_experiment(name=experiment_name)
        else:
            # 当該Experimentが存在するとき、IDを取得
            experiment_id = experiment.experiment_id

        self.run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name, description="")

        logger.info(
            f"Start MLflow Tracking: experiment_name={experiment_name} run_name={run_name} experiment_id: {experiment_id} run_id: {self.run.info.run_id}"
        )
        self.log_tag("phase", self.phase.capitalize())
        self.log_hydra_config()

    def log_hydra_config(self):
        parameters = {
            "Optimizer": self.cfg.OPTIMIZER.NAME,
            "LR scheduler": self.cfg.LR_SCHEDULER.NAME,
            "Learning Rate": self.cfg.OPTIMIZER.LR,
            "Epoch": self.cfg.EPOCH,
            "Model": self.cfg.MODEL.NAME,
            "Dataset": self.cfg.DATASET.NAME,
            "Loss": self.cfg.LOSS.NAME,
            "Input size": self.cfg.MODEL.get("INPUT_SIZE"),
            "Batch": self.cfg.BATCH,
            "GPU Ids": self.cfg.GPU.USE,
            "hostname": os.uname()[1],
        }
        mlflow.log_params(parameters)

    def log_tag(self, key, value):
        if self.use_mlflow:
            mlflow.set_tag(key, value)

    def log_metric(self, name, metric, step):
        if self.use_mlflow:
            mlflow.log_metric(f"{name}_{self.phase}", metric, step)

    def log_metrics(self, metrics: dict, step: int):
        self.last_epoch = step
        mlflow_metrics = {}
        for name, value in metrics.items():
            if self.use_mlflow:
                if isinstance(value, (int, float)):
                    mlflow_metrics[f"{name}_{self.phase}"] = value

            # Log value history
            if name not in self.histories:
                self.histories[name] = {}
            if self.phase not in self.histories[name]:
                self.histories[name][self.phase] = []
            self.histories[name][self.phase].append(value)
        mlflow.log_metrics(mlflow_metrics, step)

    def log_figure(self, fig, path):
        if self.use_mlflow:
            mlflow.log_figure(fig, path)

    def log_artifact(self, path):
        if self.use_mlflow:
            mlflow.log_artifact(path)

    def log_artifacts(self, path):
        if self.use_mlflow:
            mlflow.log_artifacts(path)

    def log_result_dir(self, path):
        """重みファイル(models以下)以外をartifactにする

        Args:
            path (str): result path
        """
        if self.use_mlflow:
            return

        for p in os.listdir(path):
            target = os.path.join(path, p)
            if os.path.isdir(target):
                if "models" in target:
                    continue
                else:
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
        if self.use_mlflow:
            mlflow.end_run(status=status)
