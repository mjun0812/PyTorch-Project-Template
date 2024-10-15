import argparse
import json
from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf

from .config import ExperimentConfig


class ConfigManager:
    description: str = "PyTorch Project"
    BASE_KEY: str = "__base__"
    IMPORT_KEY: str = "__import__"

    @staticmethod
    def argparse(func):
        def decorator(*args, **kwargs):
            cfg = ConfigManager.build()

            return func(cfg)

        return decorator

    @classmethod
    def build(cls):
        partial_parser = argparse.ArgumentParser(description=cls.description)
        partial_parser.add_argument("config", help="config file path")
        args, override_args = partial_parser.parse_known_args()

        # From Dataclass
        cfg = OmegaConf.structured(ExperimentConfig)
        ConfigManager.recursive_set_struct(cfg, False)
        # From File
        cfg_from_file = ConfigManager.build_config_from_file(args.config)
        cfg = cls.merge(cfg, cfg_from_file)
        # From CLI
        cfg_cli = OmegaConf.from_cli(override_args)
        cfg = cls.merge(cfg, cfg_cli)
        return cfg

    @staticmethod
    def recursive_set_struct(cfg: DictConfig, value: bool):
        OmegaConf.set_struct(cfg, value)
        for key, val in cfg.items():
            if isinstance(val, (DictConfig, dict)):
                ConfigManager.recursive_set_struct(val, value)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, (DictConfig, dict)):
                        ConfigManager.recursive_set_struct(item, value)

    @staticmethod
    def build_config_from_file(filename: str | Path):
        if Path(filename).suffix == ".json":
            cfg_dict = ConfigManager._load_json(filename)
        elif Path(filename).suffix in [".yaml", ".yml"]:
            cfg_dict = ConfigManager._load_yaml(filename)
        else:
            raise NotImplementedError(f"Not Supported file format: '{filename}'")
        return ConfigManager._load_base_config(cfg_dict, root_path=filename)

    @staticmethod
    def _load_json(filename: str | Path) -> DictConfig:
        with open(filename) as f:
            cfg_dict = json.load(f)
        cfg_om = OmegaConf.create(cfg_dict)
        return cfg_om

    @staticmethod
    def _load_yaml(filename: str | Path) -> DictConfig:
        cfg_om = OmegaConf.load(filename)
        return cfg_om

    @staticmethod
    def from_dict(data: dict):
        cfg = OmegaConf.create(data)
        return ConfigManager._load_base_config(cfg)

    @classmethod
    def _load_base_config(cls, cfg: DictConfig, root_path: str = None) -> DictConfig:
        if cls.BASE_KEY in cfg:
            base_cfg_path = Path(cfg.pop(cls.BASE_KEY))
            if root_path:
                base_cfg_path = Path(root_path).resolve().parent / base_cfg_path
            base_cfg = ConfigManager.build_config_from_file(base_cfg_path)
            cfg = ConfigManager.merge(base_cfg, cfg)

        # Process __import__ keys at the top level
        for key in list(cfg.keys()):  # Create a list to avoid modifying during iteration
            if isinstance(cfg[key], (dict, DictConfig)) and cls.IMPORT_KEY in cfg[key]:
                import_cfg_path = Path(cfg[key].pop(cls.IMPORT_KEY))
                if root_path:
                    import_cfg_path = Path(root_path).resolve().parent / import_cfg_path
                import_cfg = OmegaConf.load(import_cfg_path)
                cfg[key] = ConfigManager.merge(import_cfg, cfg[key])

        # Recursively process nested dictionaries
        for key, value in cfg.items():
            if isinstance(value, dict):
                cfg[key] = cls._load_base_config(value, root_path)
        return cfg

    @staticmethod
    def merge(base_dict: DictConfig, override_dict: DictConfig) -> DictConfig:
        """
        merge dict `override_dict` into `base_dict`, if the key overlapped, set replace = True to
        use the key in `a` otherwise use the key in `b`
        """
        return OmegaConf.merge(base_dict, override_dict)

    @staticmethod
    def dump(cfg: DictConfig, filename: str) -> None:
        OmegaConf.save(cfg, filename, resolve=True)

    @staticmethod
    def pretty_text(text: DictConfig, output_format: Literal["yaml", "json"] = "yaml") -> str:
        if output_format == "yaml":
            return OmegaConf.to_yaml(text, resolve=True)
        elif output_format == "json":
            return json.dumps(OmegaConf.to_object(text), indent=4, sort_keys=False)

    @staticmethod
    def to_object(cfg: DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
