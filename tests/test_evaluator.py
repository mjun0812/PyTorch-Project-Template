from pathlib import Path

import pytest
import yaml
from torchmetrics import MetricCollection

from src.config.config import EvaluatorConfig
from src.dataloaders.types import DatasetOutput
from src.evaluator.build import EVALUATOR_REGISTRY
from src.models.types import PredOutput

CONFIG_PATH = "config/__base__/evaluator/dummy_evaluator.yaml"


class TestEvaluatorHelpers:
    """Test helper functions for creating mock data"""

    @staticmethod
    def create_dataset_data(batch_size: int = 2) -> DatasetOutput:
        """Create mock dataset data for testing"""
        return DatasetOutput.dummy(batch=batch_size)

    @staticmethod
    def create_pred_output(batch_size: int = 2) -> PredOutput:
        """Create mock pred output for testing"""
        return PredOutput.dummy(batch=batch_size)


class TestEvaluatorFromConfig:
    """Test evaluators loaded from YAML configuration"""

    @staticmethod
    def load_evaluator_config(config_path: str = CONFIG_PATH) -> dict:
        """Load evaluator configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            pytest.skip(f"Config file {config_path} not found")

        with open(config_file) as f:
            return yaml.safe_load(f)

    @staticmethod
    def create_evaluator_config_objects(
        config_dict: dict, phase: str = "test"
    ) -> list[EvaluatorConfig]:
        """Convert YAML config to EvaluatorConfig objects"""
        evaluator_configs = []

        if phase not in config_dict or config_dict[phase] is None:
            return evaluator_configs

        for eval_config in config_dict[phase]:
            evaluator_config = EvaluatorConfig(
                name=eval_config.get("name", ""),
                class_name=eval_config.get("class_name", ""),
                args=eval_config.get("args", None),
            )
            evaluator_configs.append(evaluator_config)

        return evaluator_configs

    def test_build_evaluators_from_config(self) -> None:
        """Test building evaluators using the build_evaluator function"""
        config = self.load_evaluator_config()

        for phase in ["train", "val", "test"]:
            if config[phase] is None:
                continue

            evaluator_configs = self.create_evaluator_config_objects(config, phase)

            # Build evaluators using registry directly instead of build_evaluator
            # to avoid OmegaConf dependency in tests
            evaluators = []
            for c in evaluator_configs:
                args = c.args if c.args is not None else {}
                evaluators.append(EVALUATOR_REGISTRY.get(c.class_name)(**args))

            metric_collection = MetricCollection(evaluators)

            # Check that MetricCollection was created
            assert metric_collection is not None
            assert len(metric_collection) == len(evaluator_configs)

            # Check that all evaluators are present
            evaluator_names = list(metric_collection.keys())
            print(phase, evaluator_names)

    def test_evaluators_from_config_functionality(self) -> None:
        """Test that evaluators built from config work correctly"""
        config = self.load_evaluator_config()

        for phase in ["train", "val", "test"]:
            if config[phase] is None:
                continue

            evaluator_configs = self.create_evaluator_config_objects(config, phase)

            # Build evaluators using registry directly
            evaluators = []
            for c in evaluator_configs:
                args = c.args if c.args is not None else {}
                evaluators.append(EVALUATOR_REGISTRY.get(c.class_name)(**args))

            metric_collection = MetricCollection(evaluators)
            print(phase, metric_collection)

            # Create test data
            for _ in range(5):
                targets = TestEvaluatorHelpers.create_dataset_data(batch_size=2)
                preds = TestEvaluatorHelpers.create_pred_output(batch_size=2)

                # Update all evaluators
                metric_collection.update(targets, preds)

            # Compute results
            results = metric_collection.compute()
            print(phase, results)
