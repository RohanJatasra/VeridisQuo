"""Configuration service for VeridisQuo deepfake detection system.

Provides unified configuration management for the entire system.
"""

from typing import Dict, Any
from dataclasses import asdict

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert ConfigService to dictionary for JSON serialization.
        Returns:
            dict: configuration as dictionary
        """
        return {
            'inference': asdict(self.inference)
        }


__all__ = [
    'ConfigService',
]
