"""
config_loader.py

This module contains the ConfigLoader class, which is responsible for
loading and validating the YAML configuration file for the MLOps workflow
focusing on OpenAI fine-tuning.

Usage example:
    loader = ConfigLoader("config.yaml")
    config = loader.load_config()
    print(config)

Requirements:
    - pyyaml >= 5.4
"""

import os
import yaml
from typing import Any, Dict


class ConfigLoader:
    """
    ConfigLoader is responsible for loading and validating the project's
    main configuration from a specified YAML file.

    Attributes:
        config_path (str): The path to the YAML configuration file.

    Methods:
        load_config() -> Dict[str, Any]:
            Loads and validates the configuration from the YAML file,
            returning it as a Python dictionary.

        _validate_schema(config: Dict[str, Any]) -> None:
            Validates the configuration dictionary against required
            fields and logical constraints.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the ConfigLoader with the given path to the YAML file.

        Args:
            config_path (str): The path to the configuration YAML file.
        """
        self.config_path = config_path

    def load_config(self) -> Dict[str, Any]:
        """
        Loads the configuration from the YAML file and validates it.

        Returns:
            A dictionary containing the configuration parameters.

        Raises:
            FileNotFoundError: If the YAML file is not found.
            yaml.YAMLError: If the YAML file can't be parsed.
            ValueError: If configuration validation fails.
        """
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(
                f"Configuration file '{self.config_path}' does not exist."
            )

        with open(self.config_path, "r", encoding="utf-8") as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file: {e}")

        self._validate_schema(config)
        return config

    def _validate_schema(self, config: Dict[str, Any]) -> None:
        """
        Validates the config dictionary according to the expected schema and
        logical constraints.

        Args:
            config (Dict[str, Any]): The configuration dictionary to validate.

        Raises:
            ValueError: If any required key is missing or if a logical
                       constraint is violated.
        """
        # Top-level keys expected
        required_keys = ["dataset", "fine_tuning", "evaluation", "reporting"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required top-level config key: '{key}'")

        # Example: Validate dataset structure
        dataset = config["dataset"]
        for dk in ["enable", "raw_data_path", "adapter", "split"]:
            if dk not in dataset:
                raise ValueError(f"Missing dataset config key: '{dk}'")

        # Validate split ratios sum (train+val+eval should not exceed 1.0 in general)
        train_ratio = dataset["split"].get("train_ratio", 0.0)
        val_ratio = dataset["split"].get("validation_ratio", 0.0)
        eval_ratio = dataset["split"].get("evaluation_ratio", 0.0)

        if (train_ratio + val_ratio + eval_ratio) > 1.0:
            raise ValueError(
                "The sum of train_ratio, validation_ratio, and evaluation_ratio "
                "cannot exceed 1.0."
            )

        # Fine-tuning keys
        fine_tuning = config["fine_tuning"]
        for ft in ["enable", "training_file_id", "base_model", "method"]:
            if ft not in fine_tuning:
                raise ValueError(f"Missing fine_tuning config key: '{ft}'")

        # Evaluation keys
        evaluation = config["evaluation"]
        for ev in ["enable", "use_evals_api", "analysis_method"]:
            if ev not in evaluation:
                raise ValueError(f"Missing evaluation config key: '{ev}'")

        # Reporting keys
        reporting = config["reporting"]
        for rp in ["enable", "languages", "metrics", "output_format"]:
            if rp not in reporting:
                raise ValueError(f"Missing reporting config key: '{rp}'")

        # If we reach this point, config is considered valid
        # but additional logical or cross-key checks can be added as needed.
