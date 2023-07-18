import abc
import argparse
import json
from distutils.util import strtobool

import yaml
from addict import Dict

BASE_KEY = "__BASE__"


class Config:
    @staticmethod
    def main(func):
        def decorator(*args, **kwargs):
            cfg = Config.build()

            return func(cfg)

        return decorator

    @staticmethod
    def build():
        description = "PyTorch Project"
        partial_parser = argparse.ArgumentParser(description=description)
        partial_parser.add_argument("config", help="config file path")
        cfg = Config.build_config(partial_parser.parse_known_args()[0].config)

        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("config", help="config file path")
        parser = Config.add_args(parser, cfg)
        args = vars(parser.parse_args())
        args.pop("config")
        cfg = Config.merge_args_to_cfg(cfg, args)

        return cfg

    @staticmethod
    def add_args(parser, cfg: dict, prefix: str = ""):
        for k, v in cfg.items():
            if isinstance(v, str):
                parser.add_argument("--" + prefix + k)
            elif isinstance(v, bool):
                parser.add_argument("--" + prefix + k, type=strtobool, default=None)
            elif isinstance(v, int):
                parser.add_argument("--" + prefix + k, type=int)
            elif isinstance(v, float):
                parser.add_argument("--" + prefix + k, type=float)
            elif isinstance(v, dict):
                Config.add_args(parser, v, prefix + k + ".")
            elif isinstance(v, abc.Iterable):
                parser.add_argument("--" + prefix + k, type=type(next(iter(v))), nargs="+")
        return parser

    @staticmethod
    def build_config(filename: str):
        override_dict = Config._load_cfg(filename)
        return Config.load_base_config(override_dict)

    @staticmethod
    def dump(cfg: Dict, filename: str):
        with open(filename, "w") as f:
            yaml.dump(cfg, f)

    @staticmethod
    def _load_cfg(filename: str):
        if filename.endswith(".json"):
            cfg_dict = Config._load_json(filename)
        elif filename.endswith(".yaml"):
            cfg_dict = Config._load_yaml(filename)
        else:
            raise NotImplementedError(f"Not Supported file format: '{filename}'")

        cfg_dict = Dict(cfg_dict)
        return cfg_dict

    @staticmethod
    def _load_json(filename: str):
        with open(filename) as f:
            cfg_dict = json.load(f)
        return cfg_dict

    @staticmethod
    def _load_yaml(filename: str):
        with open(filename) as f:
            cfg_dict = yaml.safe_load(f)
        return cfg_dict

    @staticmethod
    def from_dict(data: dict):
        cfg = Dict(data)
        return Config.load_base_config(cfg)

    @staticmethod
    def load_base_config(cfg: Dict):
        if BASE_KEY in cfg:
            base_list = cfg.pop(BASE_KEY)
            for base_cfg_path in base_list:
                # BASEとするdictをConfigのどのKey以下に配置するかを決定する
                # "config/__BASE__/config.yaml"の中身はConfigの直下に，
                # "config/__BASE__/OPTIMIZER/Momentum.yaml"の中身はConfig.OPTIMIZER以下に
                # 配置される
                base_key = None
                path_split = base_cfg_path.split("/")
                if BASE_KEY in path_split:
                    base_path_index = path_split.index(BASE_KEY)
                    if base_path_index + 2 < len(path_split):
                        base_key = path_split[base_path_index + 1]

                if base_key:
                    base_cfg_dict = Dict({base_key: Config.build_config(base_cfg_path)})
                else:
                    base_cfg_dict = Config.build_config(base_cfg_path)

                # Merge base into current
                cfg = Config.merge_dict(base_cfg_dict, cfg, replace=True)
        return cfg

    @staticmethod
    def pretty_text(text: Dict, output_format: str = "yaml"):
        if output_format == "yaml":
            return yaml.dump(text.to_dict(), sort_keys=False)
        elif output_format == "json":
            return json.dumps(text, indent=4, sort_keys=False)

    @staticmethod
    def merge_dict(base_dict: Dict, override_dict: Dict, replace: bool = True):
        """
        merge dict `override_dict` into `base_dict`, if the key overlapped, set replace = True to
        use the key in `a` otherwise use the key in `b`
        """
        for k, v in override_dict.items():
            # if isinstance(v, dict) and k in base_dict and base_dict[k] is not None:
            if isinstance(v, dict) and k in base_dict:
                # Merge dict
                if not isinstance(base_dict[k], dict):
                    raise TypeError(
                        f"Error occured when trying to merge"
                        f"{type(base_dict[k])} with {type(v)}"
                    )
                else:
                    Config.merge_dict(base_dict[k], v, replace)
            else:
                # Direct copy
                if k in base_dict:
                    base_dict[k] = v if replace else base_dict[k]
                else:
                    base_dict[k] = v

        return base_dict

    @staticmethod
    def merge_args_to_cfg(cfg, args):
        for k, v in args.items():
            if v is None:
                continue
            elif "." in k:
                base_key, child_keys = k.split(".", 1)
                cfg[base_key] = Config.merge_args_to_cfg(cfg[base_key], {child_keys: v})
            else:
                cfg[k] = v
        return cfg
