import datetime
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib
import numpy as np
import requests
import torch
from dotenv import load_dotenv

from ..types import PathLike

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt


def get_git_hash() -> str:
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


def get_cmd() -> str:
    """実行コマンドを取得する

    Returns:
        string: 実行コマンド

    Examples:
        get_cmd()
        -> python hoge.py --huga
    """
    cmd = "python " + " ".join(sys.argv)
    return cmd


def make_output_dirs(
    output_base_path: str | Path,
    prefix: str | None = None,
    child_dirs: list[str] | None = None,
) -> Path:
    """Mkdir YYYYMMDD_HHmmSS (+ _prefix)

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
    output_path = Path(output_base_path) / f"{today}{prefix}"
    output_path.mkdir(parents=True, exist_ok=True)
    if child_dirs:
        for d in child_dirs:
            (output_path / d).mkdir(exist_ok=True, parents=True)
    return output_path


def make_result_dirs(base_path: PathLike, prefix: str = "") -> Path:
    """
    Creates a directory for storing results.

    Args:
        base_path (PathLike): The base path where the directory will be created.
        prefix (str, optional): The prefix to be added to the directory name. Defaults to "".

    Returns:
        Path: The path of the created directory.
    """
    dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        dir_name = "_" + prefix
    output_dir = Path(base_path) / "runs" / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_graph(title: str, labels: list[str], data: list, output_dir: PathLike) -> None:
    plt.gcf().clear()

    # 論文用にFontを変更する
    font_manager.fontManager.addfont("./etc/Times_New_Roman.ttf")
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 18})
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True, subplot_kw=dict(title=title))
    for label, d in zip(labels, data):
        ax.plot(d, label=label)
    ax.legend()
    fig.savefig(output_dir, dpi=100)
    plt.close()


def plot_multi_graph(filename: str, titles: list[str], data: dict, dpi: int = 300) -> None:
    font_manager.fontManager.addfont("./etc/Times_New_Roman.ttf")
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})
    col = len(titles)
    # 比率が合わないときはfigsizeをいじる
    _, axs = plt.subplots(1, col, figsize=(col * 5, 5))
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


def post_slack(channel: str = "#通知", username: str = "通知", message: str = "") -> None:
    load_dotenv(dotenv_path=f"{os.environ['HOME']}/.env")
    load_dotenv()
    token = os.getenv("SLACK_TOKEN")
    if token:
        _post_slack(token, channel, username, message)


def _post_slack(
    token: str, channel: str = "#通知", username: str = "通知", message: str = ""
) -> int:
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


class JsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
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
            return super().default(obj)


def import_submodules(module: ModuleType) -> list[ModuleType]:
    return [
        importlib.import_module(f"{module.__name__}.{f.stem}")
        for f in Path(module.__file__).parent.glob("[a-zA-Z0-9]*.py")
    ]


class HidePrints:
    """標準出力を無効にする

    Example:
        ```python
        with HidePrints():
            print("aaaa")
        ```
    """

    def __init__(self) -> None:
        self.stdout = None

    def __enter__(self) -> None:
        self.stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, ex_type: Any, ex_value: Any, trace: Any) -> None:
        sys.stdout = self.stdout


def create_symlink(target: PathLike, dst: PathLike) -> None:
    """Creates a symbolic link from the target path to the destination path.

    Both paths are relative to the current working directory.
    The symbolic link is created using a path relative to the destination.

    Args:
        target (Union[str, Path]): The target path to create a symbolic link to,
                                relative to the current directory.
        dst (Union[str, Path]): The destination path where the symbolic link will be created,
                                relative to the current directory.

    Returns:
        None
    """
    # Convert input paths to Path objects and make them absolute
    current_dir = Path.cwd()
    abs_target = current_dir / Path(target)
    abs_dst = current_dir / Path(dst)

    # Calculate the relative path from dst to target
    relative_target = abs_target.relative_to(abs_dst.parent)

    # Remove existing symlink if it exists
    if abs_dst.is_symlink():
        abs_dst.unlink()

    # Create the symlink
    abs_dst.symlink_to(relative_target)


def get_shm_size() -> int:
    stats = os.statvfs("/dev/shm")
    shm_bytes = stats.f_bsize * stats.f_blocks
    return shm_bytes


def get_free_shm_size() -> int:
    stats = os.statvfs("/dev/shm")
    free_shm_bytes = stats.f_bsize * stats.f_bavail
    return free_shm_bytes


def filter_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """クラスの__init__メソッドに必要な引数だけをkwargsから抽出する"""
    # クラスの__init__メソッドのシグネチャを取得
    sig = inspect.signature(cls.__init__)

    # 有効なパラメータ名を取得（selfを除く）
    valid_params = set(sig.parameters.keys()) - {"self"}

    # kwargsから有効なパラメータのみを抽出
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return filtered_kwargs
