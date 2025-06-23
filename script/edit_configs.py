import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description="Edit Config")
    parser.add_argument("config_path_or_dir", type=str, help="Path to the config file or dir")
    parser.add_argument(
        "override", type=str, help="Override the config file ex: batch=2,gpu.device=cpu"
    )
    args = parser.parse_args()

    # コンフィグファイルのパスを取得
    config_path = Path(args.config_path_or_dir)
    if config_path.is_dir():
        config_paths = list(config_path.rglob("*.yaml"))
    elif not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        config_paths = [config_path]

    for config_path in config_paths:
        print(f"Edit config: {config_path}")
        cfg = OmegaConf.load(config_path)

        for o in args.override.split(","):
            key, value = o.split("=")
            key = key.strip()
            value = value.strip()

            if value.lower() in ["true", "false"]:
                value = bool(value)
            elif value.isdigit():
                value = int(value)
            elif value in ["None", "none", "null"]:
                value = None
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass

            target = OmegaConf.select(cfg, key)
            if target is None:
                print(f"Key not found: {key}, Add new key")

            OmegaConf.update(cfg, key, value, merge=True)
        OmegaConf.save(cfg, config_path, resolve=False)


if __name__ == "__main__":
    main()
