import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib import font_manager
from omegaconf import OmegaConf

from src.config import ConfigManager, ExperimentConfig

matplotlib.use("Agg")
# 論文用にFontを変更する
font_manager.fontManager.addfont("./etc/Times_New_Roman.ttf")
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 18})


def _load_config(path: str) -> ExperimentConfig:
    # From Dataclass
    cfg = OmegaConf.structured(ExperimentConfig)
    # From File
    cfg_from_file = ConfigManager.build_config_from_file(path)
    cfg = ConfigManager.merge(cfg, cfg_from_file)
    return cfg


def test_config() -> None:
    cfg = _load_config(Path(__file__).parent.parent / "config/dummy.yaml")
    print(ConfigManager.pretty_text(cfg))


def test_train_test_resume() -> None:
    common_options = [
        "mlflow.use=false",
        "output=/tmp/test",
        "epoch=2",
        "val_interval=1",
        "use_ram_cache=false",
    ]
    if torch.cuda.is_available():
        common_options.append("use.gpu=0")
    else:
        common_options.append("use_cpu=true")
    print("start training test")
    cmd = ["python", "train.py", "config/dummy.yaml", *common_options]
    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        check=False,
    )
    assert process.returncode == 0
    output = process.stdout.decode("utf-8")
    print("end training test")

    output_cmd = None
    output_rows = output.split("\n")
    output_rows.reverse()
    for line in output_rows:
        if "test cmd:" in line:
            output_cmd = line.split("test cmd: ")[1].strip()
            break
    assert output_cmd is not None

    print("start test test")
    print("test cmd:", output_cmd)
    process = subprocess.run(
        output_cmd.split(" "),
        stdout=subprocess.PIPE,
        check=False,
    )
    assert process.returncode == 0
    print("end test test")

    print("start resume test")
    cmd = output_cmd.replace("test.py", "train.py").split(" ")
    process = subprocess.run(
        [*cmd, "epoch=3"],
        stdout=subprocess.PIPE,
        check=False,
    )
    assert process.returncode == 0
    print("end resume test")


def test_scripts() -> None:
    common_options = [
        Path(__file__).parent.parent / "config/dummy.yaml",
        "use_cpu=true",
        "use_ram_cache=false",
    ]
    scripts = [
        "script/show_config.py",
        "script/show_model.py",
    ]

    for script in scripts:
        print("start", script)
        process = subprocess.run(
            ["python", script, *common_options],
            stdout=subprocess.PIPE,
            check=False,
        )
        assert process.returncode == 0
        print("end", script)
