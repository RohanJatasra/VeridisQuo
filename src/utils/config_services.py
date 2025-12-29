"""Configuration service for VeridisQuo deepfake detection system.

Provides unified configuration management for the entire system.
"""

from typing import Dict, Any, Union
from dataclasses import asdict
from pathlib import Path
import json

from utils.config_dataclasses import InferenceConfig


class ConfigService:
    """Configuration service for the entire VeridisQuo system.

    Unifies inference and training configurations and provides
    methods for loading/saving configurations.
    """

    def __init__(self, inference: InferenceConfig =None):
        """Initialize ConfigService.
        Parameters:
            inference (InferenceConfig, optional): inference configuration
        """
        self.inference: InferenceConfig = inference or InferenceConfig()
        """Inference configuration."""

        # Training config will be added when integrating with training system
        # self.training: TrainingConfig = training or TrainingConfig()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigService':
        """Create ConfigService from dictionary (loaded from JSON).
        Parameters:
            config_dict (dict): configuration dictionary
        Returns:
            ConfigService: constructed config service
        """
        inference_config = InferenceConfig.from_dict(config_dict.get('inference', {}))
        return cls(inference=inference_config)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'ConfigService':
        """Load ConfigService from JSON file.
        Parameters:
            json_path (str or Path): path to JSON configuration file
        Returns:
            ConfigService: constructed config service
        Raises:
            FileNotFoundError: if JSON file doesn't exist
            ValueError: if JSON is invalid
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")

        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {json_path}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert ConfigService to dictionary for JSON serialization.
        Returns:
            dict: configuration as dictionary
        """
        return {
            'inference': asdict(self.inference)
        }

    def to_json(self, json_path: Union[str, Path], indent: int = 2) -> None:
        """Save ConfigService to JSON file.
        Parameters:
            json_path (str or Path): path to JSON output file
            indent (int): JSON indentation level for readability
        Raises:
            RuntimeError: if saving fails
        """
        json_path = Path(json_path)

        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)

            with open(json_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=indent)
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration to {json_path}: {e}")


__all__ = [
    'ConfigService',
]
