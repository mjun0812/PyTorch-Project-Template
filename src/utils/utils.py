import datetime
import importlib
import json
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path

import matplotlib
import numpy as np
import requests
import torch
import torch.distributed as dist
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa
import matplotlib.pyplot as plt  # noqa

logger = logging.getLogger()


def get_git_hash():
    """gitハッシュを取得する

    Returns:
        string: Gitのハッシュ値
    """
    cmd = "git rev-parse --short HEAD"
    try:
        git_hash = (
            subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT).strip().decode("utf-8")
        )
    except Exception:
        git_hash = "Not found git repository"
    return git_hash


def get_cmd():
    """実行コマンドを取得する
    Returns:
        string: 実行コマンド

    Examples:
        get_cmd()
        -> python hoge.py --huga
    """
    cmd = "python " + " ".join(sys.argv)
    return cmd


def make_output_dirs(output_base_path: str, prefix="", child_dirs=None) -> str:
    """mkdir YYYYMMDD_HHmmSS (+ _prefix)

    Args:
        output_base_path (str): make output dir path.
        prefix (str, optional): add prefix mkdir. Defaults to "".
        child_dirs ([type], optional): mkdir child dir list. Defaults to None.

    Returns:
        str: YYYYMMDD_HHmmSS

    Examples:
    ```python
    out = make_output_dirs("./result", prefix="MODEL", child_dirs=["models", "figs"])

    ./result/21010812_120000
    ├── models
    └── figs
    ```
    """
    today = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        prefix = "_" + prefix
    output_path = os.path.join(output_base_path, f"{today}{prefix}")
    os.makedirs(output_path, exist_ok=True)
    if child_dirs:
        for d in child_dirs:
            os.makedirs(os.path.join(output_path, d), exist_ok=True)
    return output_path


def make_result_dirs(weight_path, prefix=""):
    weight_name, _ = os.path.splitext(os.path.basename(weight_path))
    today = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        prefix = "_" + prefix
    dir_name = f"{today}_{weight_name}{prefix}"
    output_dir = os.path.dirname(os.path.dirname(weight_path))
    output_dir = os.path.join(output_dir, "runs", dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_graph(title, labels, data, output_dir):
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
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True, subplot_kw=dict(title=title))
    for label, d in zip(labels, data):
        ax.plot(d, label=label)
    ax.legend()
    fig.savefig(output_dir, dpi=100)
    plt.close()


def plot_multi_graph(filename, titles, data, dpi=300):
    font_manager.fontManager.addfont("./etc/Times_New_Roman.ttf")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            # "text.usetex": True,
            "ps.useafm": True,
            "pdf.use14corefonts": True,
        }
    )
    col = len(titles)
    # 比率が合わないときはfigsizeをいじる
    fig, axs = plt.subplots(1, col, figsize=(col * 5, 5))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # 1次元にしてforで回せるように。行->列の順
    a = axs.ravel()
    for i, title in enumerate(titles):
        plot_data = data[title]
        for plot in plot_data:
            a[i].plot(plot["data"], label=plot["label"])
        a[i].legend()
        a[i].grid()
        a[i].set_title(title)  # 上にキャプション

    # DPIがでかすぎるとファイルサイズも大きくなり、プログラムの速度も落ちる
    # DPI * figsizeの解像度の画像ができる
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.gcf().clear()
    plt.close()


def post_slack(channel="#通知", username="通知", message=""):
    load_dotenv(dotenv_path=f"{os.environ['HOME']}/.env")
    load_dotenv()
    token = os.getenv("SLACK_TOKEN")
    if token:
        _post_slack(token, channel, username, message)


def _post_slack(token, channel="#通知", username="通知", message=""):
    """slackにメッセージを送る. send slack message

    Args:
        token (str): Slack Token
        channel (str, optional): メッセージを送る通知先. Defaults to "#通知".
        username (str, optional): メッセージを送るユーザーの名前. Defaults to "通知".
        message (str, optional): send message. Defaults to "".

    Returns:
        int: http status code
    """
    response = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Content-Type": "application/json"},
        params={
            "token": token,
            "channel": channel,
            "text": message,
            "username": username,
        },
    )
    return response.status_code


def error_handle(e, phase, message):
    message = (
        f"Error {phase}\n{e}\n"
        f"{traceback.format_exc()}\n"
        f"{message}\n"
        f"Host: {os.uname()[1]}"
    )
    logger.error(message)
    post_slack(channel="#error", message=message)
    if dist.is_initialized():
        dist.destroy_process_group()


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.complexfloating)):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            if obj.dim() == 0:
                return obj.item()
            else:
                return obj.cpu().tolist()
        else:
            return super(JsonEncoder, self).default(obj)


def import_submodules(module):
    return [
        importlib.import_module(f"{module.__name__}.{f.stem}")
        for f in Path(module.__file__).parent.glob("[a-zA-Z0-9]*.py")
    ]


def create_symlink(target, dst):
    """
    シンボリックリンクを作成します。

    Args:
        target (str): シンボリックリンクのターゲットとなるファイルまたはディレクトリのパス。
        dst (str): シンボリックリンクの作成先のパス。
    """
    relative_target = os.path.relpath(target, os.path.dirname(dst))
    if os.path.islink(dst):
        os.unlink(dst)
    os.symlink(relative_target, dst)
