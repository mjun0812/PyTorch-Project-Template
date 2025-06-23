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
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from dotenv import load_dotenv

from ..types import PathLike


def get_git_hash() -> str:
    """Get the current Git commit hash.

    Returns:
        Git commit hash as a short string, or error message if not in a Git repository.
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
    """Get the command line used to execute the current script.

    Returns:
        Command line string including python and all arguments.

    Examples:
        >>> get_cmd()
        'python train.py config/dummy.yaml batch=32'
    """
    cmd = "python " + " ".join(sys.argv)
    return cmd


def make_output_dirs(
    output_base_path: str | Path,
    prefix: str | None = None,
    child_dirs: list[str] | None = None,
) -> Path:
    """Create timestamped output directory with optional subdirectories.

    Args:
        output_base_path: Base path where the output directory will be created.
        prefix: Optional prefix to add to the directory name.
        child_dirs: List of subdirectory names to create.

    Returns:
        Path to the created output directory.

    Examples:
        >>> out = make_output_dirs("./result", prefix="MODEL", child_dirs=["models", "figs"])
        >>> # Creates: ./result/20240108_120000_MODEL/
        >>> #          ├── models/
        >>> #          └── figs/
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
    """Create a timestamped directory for storing experiment results.

    Args:
        base_path: The base path where the directory will be created.
        prefix: Optional prefix to be added to the directory name.

    Returns:
        Path to the created result directory.
    """
    dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        dir_name = "_" + prefix
    output_dir = Path(base_path) / "runs" / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_graph(title: str, labels: list[str], data: list, output_dir: PathLike) -> None:
    """Plot multiple data series on a single graph.

    Args:
        title: Title for the plot.
        labels: Labels for each data series.
        data: List of data series to plot.
        output_dir: Path where the plot will be saved.
    """
    matplotlib.use("Agg")

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
    """Create multiple subplots in a single figure.

    Args:
        filename: Output filename for the plot.
        titles: Titles for each subplot.
        data: Dictionary containing plot data for each title.
        dpi: Resolution for the saved figure.
    """
    matplotlib.use("Agg")
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


def post_slack(channel: str = "#notifications", username: str = "Bot", message: str = "") -> None:
    """Send a message to Slack using either token or webhook.

    Automatically detects and uses either SLACK_TOKEN or SLACK_WEBHOOK_URL
    from environment variables. Environment variables override function arguments.

    Args:
        channel: Slack channel to send the message to.
        username: Username to display as the sender.
        message: Message content to send.
    """
    load_dotenv(dotenv_path=f"{os.environ['HOME']}/.env")
    load_dotenv()
    token = os.getenv("SLACK_TOKEN")
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    env_channel = os.getenv("SLACK_CHANNEL")
    env_username = os.getenv("SLACK_USERNAME")

    # 環境変数でchannelが設定されている場合は優先
    if env_channel:
        channel = env_channel
    # 環境変数でusernameが設定されている場合は優先
    if env_username:
        username = env_username

    if token:
        _post_slack(token, channel, username, message)
    elif webhook_url:
        _post_slack_webhook(webhook_url, channel, username, message)


def _post_slack(
    token: str, channel: str = "#notifications", username: str = "Bot", message: str = ""
) -> int:
    """Send a message to Slack using API token.

    Args:
        token: Slack API token.
        channel: Slack channel to send the message to.
        username: Username to display as the sender.
        message: Message content to send.

    Returns:
        HTTP status code from the Slack API response.
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


def _post_slack_webhook(
    webhook_url: str, channel: str = "#notifications", username: str = "Bot", message: str = ""
) -> int:
    """Send a message to Slack using webhook URL.

    Args:
        webhook_url: Slack webhook URL.
        channel: Slack channel to send the message to.
        username: Username to display as the sender.
        message: Message content to send.

    Returns:
        HTTP status code from the webhook response.
    """
    payload = {
        "channel": channel,
        "username": username,
        "text": message,
    }

    response = requests.post(
        webhook_url,
        headers={"Content-Type": "application/json"},
        json=payload,
    )
    return response.status_code


class JsonEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy arrays and PyTorch tensors.

    Extends the default JSON encoder to handle NumPy arrays, NumPy scalars,
    and PyTorch tensors by converting them to Python native types.
    """

    def default(self, obj: Any) -> Any:
        """Convert special objects to JSON-serializable types.

        Args:
            obj: Object to serialize.

        Returns:
            JSON-serializable representation of the object.
        """
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
    """Import all Python submodules from a given module.

    Args:
        module: Parent module to import submodules from.

    Returns:
        List of imported submodules.
    """
    return [
        importlib.import_module(f"{module.__name__}.{f.stem}")
        for f in Path(module.__file__).parent.glob("[a-zA-Z0-9]*.py")
    ]


class HidePrints:
    """Context manager to suppress standard output.

    Example:
        >>> with HidePrints():
        ...     print("This won't be printed")
    """

    def __init__(self) -> None:
        """Initialize the context manager."""
        self.stdout = None

    def __enter__(self) -> None:
        """Enter context and suppress stdout."""
        self.stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, ex_type: Any, ex_value: Any, trace: Any) -> None:
        """Exit context and restore stdout.

        Args:
            ex_type: Exception type.
            ex_value: Exception value.
            trace: Exception traceback.
        """
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
    """Get the total size of shared memory filesystem.

    Returns:
        Total shared memory size in bytes.
    """
    stats = os.statvfs("/dev/shm")
    shm_bytes = stats.f_bsize * stats.f_blocks
    return shm_bytes


def get_free_shm_size() -> int:
    """Get the available size of shared memory filesystem.

    Returns:
        Available shared memory size in bytes.
    """
    stats = os.statvfs("/dev/shm")
    free_shm_bytes = stats.f_bsize * stats.f_bavail
    return free_shm_bytes


def filter_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs to only include arguments accepted by a class's __init__ method.

    Args:
        cls: Class to inspect for valid arguments.
        kwargs: Dictionary of keyword arguments to filter.

    Returns:
        Dictionary containing only valid arguments for the class.
    """
    # クラスの__init__メソッドのシグネチャを取得
    sig = inspect.signature(cls.__init__)

    # 有効なパラメータ名を取得（selfを除く）
    valid_params = set(sig.parameters.keys()) - {"self"}

    # kwargsから有効なパラメータのみを抽出
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return filtered_kwargs
