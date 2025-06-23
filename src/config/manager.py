import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf

from .config import ExperimentConfig


class ConfigManager:
    """Configuration manager for handling hierarchical configurations.

    This class provides utilities for building configurations from dataclasses,
    YAML/JSON files, and command-line arguments with support for base config
    inheritance and imports.

    Attributes:
        description: Description for the argument parser.
        BASE_KEY: Key used for base configuration inheritance.
        IMPORT_KEY: Key used for importing specific configuration sections.
    """

    description: str = "PyTorch Project"
    BASE_KEY: str = "__base__"
    IMPORT_KEY: str = "__import__"

    @staticmethod
    def argparse(func: Callable[[DictConfig], Any]) -> Callable[[], Any]:
        """Decorator to automatically build and pass configuration to a function.

        Args:
            func: Function that takes a DictConfig as its first argument.

        Returns:
            Decorated function that builds config automatically.
        """

        def decorator(*args: Any, **kwargs: Any) -> Any:
            cfg = ConfigManager.build()

            return func(cfg)

        return decorator

    @classmethod
    def build(cls) -> DictConfig:
        """Build configuration from dataclass defaults, file, and CLI arguments.

        The configuration is built in the following order of precedence:
        1. Dataclass defaults (lowest priority)
        2. Configuration file values
        3. Command-line arguments (highest priority)

        Returns:
            Merged configuration as DictConfig.
        """
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
    def recursive_set_struct(cfg: DictConfig, value: bool) -> None:
        """Recursively set struct mode for DictConfig and nested configurations.

        Args:
            cfg: Configuration to modify.
            value: Whether to enable struct mode.
        """
        OmegaConf.set_struct(cfg, value)
        for key, val in cfg.items():
            if isinstance(val, (DictConfig, dict)):
                ConfigManager.recursive_set_struct(val, value)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, (DictConfig, dict)):
                        ConfigManager.recursive_set_struct(item, value)

    @staticmethod
    def build_config_from_file(filename: str | Path) -> DictConfig:
        """Build configuration from a file with base config inheritance.

        Supports YAML, YML, and JSON formats. Automatically handles base
        configuration inheritance using the __base__ key.

        Args:
            filename: Path to the configuration file.

        Returns:
            Configuration loaded from file with base configs merged.

        Raises:
            NotImplementedError: If file format is not supported.
        """
        if Path(filename).suffix == ".json":
            cfg_dict = ConfigManager._load_json(filename)
        elif Path(filename).suffix in [".yaml", ".yml"]:
            cfg_dict = ConfigManager._load_yaml(filename)
        else:
            raise NotImplementedError(f"Not Supported file format: '{filename}'")
        return ConfigManager._load_base_config(cfg_dict, root_path=filename)

    @staticmethod
    def _load_json(filename: str | Path) -> DictConfig:
        """Load configuration from a JSON file.

        Args:
            filename: Path to the JSON file.

        Returns:
            Configuration as DictConfig.
        """
        with open(filename) as f:
            cfg_dict = json.load(f)
        cfg_om = OmegaConf.create(cfg_dict)
        return cfg_om

    @staticmethod
    def _load_yaml(filename: str | Path) -> DictConfig:
        """Load configuration from a YAML file.

        Args:
            filename: Path to the YAML file.

        Returns:
            Configuration as DictConfig.
        """
        cfg_om = OmegaConf.load(filename)
        return cfg_om

    @staticmethod
    def from_dict(data: dict) -> DictConfig:
        """Create configuration from a dictionary with base config processing.

        Args:
            data: Dictionary containing configuration data.

        Returns:
            Configuration as DictConfig with base configs merged.
        """
        cfg = OmegaConf.create(data)
        return ConfigManager._load_base_config(cfg)

    @classmethod
    def _load_base_config(cls, cfg: DictConfig, root_path: str | None = None) -> DictConfig:
        """Load and merge base configurations recursively.

        Processes __base__ keys for configuration inheritance and __import__ keys
        for importing specific configuration sections.

        Args:
            cfg: Configuration to process.
            root_path: Root path for resolving relative config paths.

        Returns:
            Configuration with base configs and imports merged.
        """
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
        """Merge two configurations with override precedence.

        When keys overlap, values from override_dict take precedence over
        values from base_dict.

        Args:
            base_dict: Base configuration.
            override_dict: Configuration with values to override.

        Returns:
            Merged configuration.
        """
        return OmegaConf.merge(base_dict, override_dict)

    @staticmethod
    def dump(cfg: DictConfig, filename: str) -> None:
        """Save configuration to a file.

        Args:
            cfg: Configuration to save.
            filename: Output file path.
        """
        OmegaConf.save(cfg, filename, resolve=True)

    @staticmethod
    def pretty_text(text: DictConfig, output_format: Literal["yaml", "json"] = "yaml") -> str:
        """Convert configuration to formatted string.

        Args:
            text: Configuration to format.
            output_format: Output format (yaml or json).

        Returns:
            Formatted configuration string.
        """
        if output_format == "yaml":
            return OmegaConf.to_yaml(text, resolve=True)
        elif output_format == "json":
            return json.dumps(OmegaConf.to_object(text), indent=4, sort_keys=False)

    @staticmethod
    def to_object(cfg: DictConfig) -> object:
        """Convert DictConfig to plain Python object.

        Args:
            cfg: Configuration to convert.

        Returns:
            Configuration as plain Python object.
        """
        return OmegaConf.to_container(cfg, resolve=True)
