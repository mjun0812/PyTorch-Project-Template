import os
import datetime
import logging

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from dotenv import load_dotenv
import requests

logger = logging.getLogger()


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
    try:
        load_dotenv()
        SLACK_TOKEN = os.getenv("SLACK_TOKEN")
        response = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Content-Type": "application/json"},
            params={
                "token": SLACK_TOKEN,
                "channel": channel,
                "text": message,
                "username": username,
            },
        )
    except Exception:
        pass
    return response.status_code


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
