"""Configuration service for VeridisQuo deepfake detection system.
Provides unified configuration management for the entire system.
"""

from typing import Dict, Any, Union, overload
from pathlib import Path
import json
import yaml
from logging import Logger, getLogger


class ConfigService(Dict[str, Any]):
    """Configuration service with utilitaries for the entire VeridisQuo config system."""

    class_logger: Logger = getLogger("/".join(__file__.split("/")[-2:]))
    
    def __init__(self, config: Dict[str, Any] =None):
        super().__init__()
        self.logger: Logger = getLogger("/".join(__file__.split("/")[-2:]))
        if config:
            self.update(config)
    
    @classmethod
    def load_config(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from a JSON or YAML file.
        Automatically detects file format based on extension (.json, .yml, .yaml).
        Parameters:
            config_path (str or Path): path to the configuration file
        Returns:
            Dict[str, Any]: configuration dictionary
        Raises:
            RuntimeError: if file not found or format unsupported
        """
        try:
            config_path = Path(config_path)

            assert config_path.exists(), f"Configuration file not found: {config_path}"

            file_extension = config_path.suffix.lower()
            with open(config_path, 'r', encoding='utf-8') as f:
                file_content = f.read()

            if file_extension == '.json':
                config = cls.load_config(file_content)
            elif file_extension in ['.yml', '.yaml']:
                config = cls.load_config(file_content)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_extension}. Supported formats: .json, .yml, .yaml")

            cls.logger.info(f"Configuration loaded successfully from {config_path} ({file_extension[1:].upper()})")
            return config

        except (AssertionError, FileNotFoundError, ValueError) as e:
            err_mess = f"Error loading configuration from {config_path}: {e}"
            cls.logger.fatal(err_mess)
            raise RuntimeError(err_mess) from e
     
    @classmethod
    @overload
    def load_config(cls, json_data: str) -> Dict[str, Any]:
        """Load configuration from a JSON string.
        Parameters:
            json_data (str): JSON string containing configuration
        Returns:
            Dict[str, Any]: configuration dictionary
        Raises:
            ValueError: if JSON is invalid or empty
        """
        try:
            config = json.loads(json_data)

            assert isinstance(config, dict), "Configuration JSON must represent an object"
            assert config, "Configuration JSON is empty"

            cls.logger.info("Configuration loaded successfully from JSON string")
            return config

        except (json.JSONDecodeError, AssertionError) as e:
            err_mess = f"Invalid JSON data: {e}"
            cls.logger.fatal(err_mess)
            raise ValueError(err_mess) from e
    
    @classmethod
    @overload
    def load_config(cls, yaml_data: str) -> Dict[str, Any]:
        """Load configuration from a YAML string.
        Parameters:
            yaml_data (str): YAML string containing configuration
        Returns:
            Dict[str, Any]: configuration dictionary
        Raises:
            ValueError: if YAML is invalid or empty
        """
        try:
            config = yaml.safe_load(yaml_data)

            assert isinstance(config, dict), "Configuration YAML must represent a mapping"
            assert config, "Configuration YAML is empty"

            cls.logger.info("Configuration loaded successfully from YAML string")
            return config

        except (yaml.YAMLError, AssertionError) as e:
            err_mess = f"Invalid YAML data: {e}"
            cls.logger.fatal(err_mess)
            raise ValueError(err_mess) from e
 
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save the provided configuration dictionary internally.
        Parameters:
            config (Dict[str, Any]): configuration dictionary to save
        """
        self.clear()
        self.update(config)
        self.logger.info("Configuration saved internally.")
         
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update the internally saved configuration dictionary with new values.
        Parameters:
            updates (Dict[str, Any]): dictionary with updates to apply
        """
        self.update(updates)
        self.logger.info("Internally saved configuration updated.")
          
    def get_saved_config(self) -> Dict[str, Any]:
        """Retrieve the internally saved configuration dictionary.
        Returns:
            Dict[str, Any]: saved configuration dictionary
        """
        self.logger.info("Retrieved internally saved configuration.")
        return self.saved_config
    
    def validate_config(self, config: Dict[str, Any], required_keys: Dict[str, Any]) -> bool:
        """TODO: Implement configuration validation logic."""