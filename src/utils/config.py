"""Configuration loader utilities.
Provides functions to load and parse JSON configuration files for VeridisQuo.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from logging import Logger, getLogger

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load configuration from a JSON file.
    Parameters:
        config_path (str | Path): path to the JSON configuration file
    Returns:
        Dict[str, Any]: configuration dictionary
    Raises:
        FileNotFoundError: if the configuration file does not exist
        json.JSONDecodeError: if the file contains invalid JSON
        ValueError: if the configuration is empty or invalid
    """
    logger: Logger = getLogger("utils.config.load_config")

    try:
        config_path = Path(config_path)

        if not config_path.exists():
            logger.fatal(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if not config:
            logger.fatal(f"Configuration file is empty: {config_path}")
            raise ValueError(f"Configuration file is empty: {config_path}")

        logger.info(f"Configuration loaded successfully from {config_path}")
        return config

    except json.JSONDecodeError as e:
        err_mess = f"Invalid JSON in configuration file {config_path}: {e}"
        logger.fatal(err_mess)
        raise ValueError(err_mess) from e
    
    except Exception as e:
        err_mess = f"Error loading configuration from {config_path}: {e}"
        logger.fatal(err_mess)
        raise RuntimeError(err_mess) from e


def load_paths_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract paths configuration section.
    Parameters:
        config (Dict[str, Any]): full configuration dictionary
    Returns:
        Dict[str, Any]: paths configuration section
    Raises:
        KeyError: if paths section is missing from configuration
    """
    logger: Logger = getLogger("utils.config.load_paths_config")

    try:
        paths_config = config["paths"]
        logger.debug(f"Paths config loaded with {len(paths_config)} entries")
        return paths_config
    except KeyError as e:
        logger.error(f"Missing 'paths' section in configuration: {e}")
        raise KeyError("Missing 'paths' section in configuration") from e


def load_device_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract device configuration section.
    Parameters:
        config (Dict[str, Any]): full configuration dictionary
    Returns:
        Dict[str, Any]: device configuration section
    Raises:
        KeyError: if device section is missing from configuration
    """
    logger: Logger = getLogger("utils.config.load_device_config")

    try:
        device_config = config["device"]
        logger.debug(f"Device config loaded: {device_config.get('type', 'auto')}")
        return device_config
    except KeyError as e:
        logger.error(f"Missing 'device' section in configuration: {e}")
        raise KeyError("Missing 'device' section in configuration") from e


def load_model_architecture_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model architecture configuration section.
    Parameters:
        config (Dict[str, Any]): full configuration dictionary
    Returns:
        Dict[str, Any]: model architecture configuration section
    Raises:
        KeyError: if model_architecture section is missing from configuration
    """
    logger: Logger = getLogger("utils.config.load_model_architecture_config")

    try:
        arch_config = config["model_architecture"]
        logger.debug(f"Model architecture config loaded with channel_mode: {arch_config.get('channel_mode')}")
        return arch_config
    except KeyError as e:
        logger.error(f"Missing 'model_architecture' section in configuration: {e}")
        raise KeyError("Missing 'model_architecture' section in configuration") from e


def load_preprocessing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract preprocessing configuration section.
    Parameters:
        config (Dict[str, Any]): full configuration dictionary
    Returns:
        Dict[str, Any]: preprocessing configuration section
    Raises:
        KeyError: if preprocessing section is missing from configuration
    """
    logger: Logger = getLogger("utils.config.load_preprocessing_config")

    try:
        preprocessing_config = config["preprocessing"]
        logger.debug(f"Preprocessing config loaded with {len(preprocessing_config)} sections")
        return preprocessing_config
    except KeyError as e:
        logger.error(f"Missing 'preprocessing' section in configuration: {e}")
        raise KeyError("Missing 'preprocessing' section in configuration") from e


def load_inference_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract inference configuration section.
    Parameters:
        config (Dict[str, Any]): full configuration dictionary
    Returns:
        Dict[str, Any]: inference configuration section
    Raises:
        KeyError: if inference section is missing from configuration
    """
    logger: Logger = getLogger("utils.config.load_inference_config")

    try:
        inference_config = config["inference"]
        logger.debug(f"Inference config loaded with {len(inference_config)} sections")
        return inference_config
    except KeyError as e:
        logger.error(f"Missing 'inference' section in configuration: {e}")
        raise KeyError("Missing 'inference' section in configuration") from e


def load_logging_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract logging configuration section.
    Parameters:
        config (Dict[str, Any]): full configuration dictionary
    Returns:
        Dict[str, Any]: logging configuration section
    Raises:
        KeyError: if logging section is missing from configuration
    """
    logger: Logger = getLogger("utils.config.load_logging_config")

    try:
        logging_config = config["logging"]
        logger.debug(f"Logging config loaded: level={logging_config.get('level', 'INFO')}")
        return logging_config
    except KeyError as e:
        logger.error(f"Missing 'logging' section in configuration: {e}")
        raise KeyError("Missing 'logging' section in configuration") from e


def load_performance_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract performance configuration section.
    Parameters:
        config (Dict[str, Any]): full configuration dictionary
    Returns:
        Dict[str, Any]: performance configuration section
    Raises:
        KeyError: if performance section is missing from configuration
    """
    logger: Logger = getLogger("utils.config.load_performance_config")

    try:
        performance_config = config["performance"]
        logger.debug(f"Performance config loaded with {len(performance_config)} sections")
        return performance_config
    except KeyError as e:
        logger.error(f"Missing 'performance' section in configuration: {e}")
        raise KeyError("Missing 'performance' section in configuration") from e


def load_production_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract production configuration section.
    Parameters:
        config (Dict[str, Any]): full configuration dictionary
    Returns:
        Dict[str, Any]: production configuration section
    Raises:
        KeyError: if production section is missing from configuration
    """
    logger: Logger = getLogger("utils.config.load_production_config")

    try:
        production_config = config["production"]
        logger.debug(f"Production config loaded: API enabled={production_config.get('api', {}).get('enabled', False)}")
        return production_config
    except KeyError as e:
        logger.error(f"Missing 'production' section in configuration: {e}")
        raise KeyError("Missing 'production' section in configuration") from e


def load_frequency_features_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract frequency features configuration from model architecture.
    Parameters:
        config (Dict[str, Any]): full configuration dictionary
    Returns:
        Dict[str, Any]: frequency features configuration section
    Raises:
        KeyError: if frequency_features section is missing from model_architecture
    """
    logger: Logger = getLogger("utils.config.load_frequency_features_config")

    try:
        arch_config = config["model_architecture"]
        freq_config = arch_config["frequency_features"]
        logger.debug(f"Frequency features config loaded: use_gpu={freq_config.get('use_gpu_extractors', False)}")
        return freq_config
    except KeyError as e:
        logger.error(f"Missing 'frequency_features' in model_architecture: {e}")
        raise KeyError("Missing 'frequency_features' in model_architecture") from e


def load_spatial_features_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract spatial features configuration from model architecture.
    Parameters:
        config (Dict[str, Any]): full configuration dictionary
    Returns:
        Dict[str, Any]: spatial features configuration section
    Raises:
        KeyError: if spatial_features section is missing from model_architecture
    """
    logger: Logger = getLogger("utils.config.load_spatial_features_config")

    try:
        arch_config = config["model_architecture"]
        spatial_config = arch_config["spatial_features"]
        logger.debug(f"Spatial features config loaded: {spatial_config.get('name', 'N/A')}")
        return spatial_config
    except KeyError as e:
        logger.error(f"Missing 'spatial_features' in model_architecture: {e}")
        raise KeyError("Missing 'spatial_features' in model_architecture") from e


def load_classifier_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract classifier configuration from model architecture.
    Parameters:
        config (Dict[str, Any]): full configuration dictionary
    Returns:
        Dict[str, Any]: classifier configuration section
    Raises:
        KeyError: if classifier section is missing from model_architecture
    """
    logger: Logger = getLogger("utils.config.load_classifier_config")

    try:
        arch_config = config["model_architecture"]
        classifier_config = arch_config["classifier"]
        logger.debug(f"Classifier config loaded: input_dim={classifier_config.get('input_dim')}")
        return classifier_config
    except KeyError as e:
        logger.error(f"Missing 'classifier' in model_architecture: {e}")
        raise KeyError("Missing 'classifier' in model_architecture") from e


def load_face_detection_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract face detection configuration from preprocessing.
    Parameters:
        config (Dict[str, Any]): full configuration dictionary
    Returns:
        Dict[str, Any]: face detection configuration section
    Raises:
        KeyError: if face_detection section is missing from preprocessing
    """
    logger: Logger = getLogger("utils.config.load_face_detection_config")

    try:
        preprocessing_config = config["preprocessing"]
        face_detection_config = preprocessing_config["face_detection"]
        logger.debug(f"Face detection config loaded: enabled={face_detection_config.get('enabled', False)}")
        return face_detection_config
    except KeyError as e:
        logger.error(f"Missing 'face_detection' in preprocessing: {e}")
        raise KeyError("Missing 'face_detection' in preprocessing") from e


def load_frame_extraction_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract frame extraction configuration from preprocessing.
    Parameters:
        config (Dict[str, Any]): full configuration dictionary
    Returns:
        Dict[str, Any]: frame extraction configuration section
    Raises:
        KeyError: if frame_extraction section is missing from preprocessing
    """
    logger: Logger = getLogger("utils.config.load_frame_extraction_config")

    try:
        preprocessing_config = config["preprocessing"]
        frame_extraction_config = preprocessing_config["frame_extraction"]
        logger.debug(f"Frame extraction config loaded: fps={frame_extraction_config.get('frames_per_second')}")
        return frame_extraction_config
    except KeyError as e:
        logger.error(f"Missing 'frame_extraction' in preprocessing: {e}")
        raise KeyError("Missing 'frame_extraction' in preprocessing") from e
