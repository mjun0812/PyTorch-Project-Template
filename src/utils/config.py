import argparse
import json

from omegaconf import DictConfig, OmegaConf

BASE_KEY = "__BASE__"
BASE_DATASET_KEY = "__DATASET__"


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
        args, override_args = partial_parser.parse_known_args()
        cfg = Config.build_config(args.config)

        cfg_cli = OmegaConf.from_cli(override_args)
        cfg_cli = Config.load_base_config(cfg_cli)

        cfg = Config.merge_dict(cfg, cfg_cli)

        return cfg

    @staticmethod
    def build_config(filename: str):
        override_dict = Config._load_cfg(filename)
        return Config.load_base_config(override_dict)

    @staticmethod
    def dump(cfg: DictConfig, filename: str):
        OmegaConf.save(cfg, filename, resolve=True)

    @staticmethod
    def _load_cfg(filename: str):
        if filename.endswith(".json"):
            cfg_dict = Config._load_json(filename)
            cfg_dict = OmegaConf.create(cfg_dict)
        elif filename.endswith(".yaml"):
            cfg_dict = Config._load_yaml(filename)
        else:
            raise NotImplementedError(f"Not Supported file format: '{filename}'")

        return cfg_dict

    @staticmethod
    def _load_json(filename: str):
        with open(filename) as f:
            cfg_dict = json.load(f)
        return cfg_dict

    @staticmethod
    def _load_yaml(filename: str):
        cfg_dict = OmegaConf.load(filename)
        return cfg_dict

    @staticmethod
    def from_dict(data: dict):
        cfg = OmegaConf.create(data)
        return Config.load_base_config(cfg)

    @staticmethod
    def load_base_config(cfg: DictConfig):
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
                    base_cfg_dict = OmegaConf.create(
                        {base_key: Config.build_config(base_cfg_path)}
                    )
                else:
                    base_cfg_dict = Config.build_config(base_cfg_path)

                # Merge base into current
                cfg = Config.merge_dict(base_cfg_dict, cfg)
        if BASE_DATASET_KEY in cfg:
            base_cfg_dict = OmegaConf.create(
                {"DATASET": Config.build_config(cfg.pop(BASE_DATASET_KEY))}
            )
            # Merge base into current
            cfg = Config.merge_dict(base_cfg_dict, cfg)
        return cfg

    @staticmethod
    def pretty_text(text: DictConfig, output_format: str = "yaml"):
        if output_format == "yaml":
            return OmegaConf.to_yaml(text, resolve=True)
        elif output_format == "json":
            return json.dumps(OmegaConf.to_object(text), indent=4, sort_keys=False)

    @staticmethod
    def merge_dict(base_dict: DictConfig, override_dict: DictConfig):
        """
        merge dict `override_dict` into `base_dict`, if the key overlapped, set replace = True to
        use the key in `a` otherwise use the key in `b`
        """
        # データセットは上書きではなく入れ替え(replace)を行う
        if "DATASET" in override_dict:
            base_dict.DATASET = override_dict.DATASET
        base_dict = OmegaConf.merge(base_dict, override_dict)
        return base_dict
