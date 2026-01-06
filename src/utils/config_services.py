"""Configuration service for VeridisQuo deepfake detection system.
Provides unified configuration management for the entire system.
"""

from typing import Dict, Any, Union, overload
from pathlib import Path
import json
import yaml
from logging import Logger, getLogger
from .config_schemas import (
    InferenceConfig, ModelArchitectureConfig, SpatialFeatureConfig, 
    FrequencyFeatureConfig, ClassifierConfig, FaceDetectionConfig, 
    FaceExtractionConfig, TrainingConfig, PreprocessingConfig, DataLoaderConfig
)

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

            cls.class_logger.info(f"Configuration loaded successfully from {config_path} ({file_extension[1:].upper()})")
            return config

        except (AssertionError, FileNotFoundError, ValueError) as e:
            err_mess = f"Error loading configuration from {config_path}: {e}"
            cls.class_logger.fatal(err_mess)
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
        return dict(self)
    
    def validate_config(self, config: Dict[str, Any], required_keys: Dict[str, type]) -> bool:
        """Validate configuration against required keys and their expected types.
        Parameters:
            config (Dict[str, Any]): configuration dictionary to validate
            required_keys (Dict[str, type]): dictionary mapping required key names to their expected types
        Returns:
            bool: True if validation passes
        Raises:
            ValueError: if validation fails (missing keys or incorrect types)
        """
        try:
            for key, expected_type in required_keys.items():
                assert key in config, f"Missing required configuration key: {key}"
                if expected_type is not None:
                    actual_value = config[key]
                    assert isinstance(actual_value, expected_type), (
                        f"Invalid type for key '{key}': expected {expected_type.__name__}, "
                        f"got {type(actual_value).__name__}"
                    )
            self.logger.info("Configuration validation passed.")
            return True
        except AssertionError as e:
            err_msg = f"Configuration validation failed: {e}"
            self.logger.error(err_msg)
            raise ValueError(err_msg) from e
    
    def extract_inference_config(self) -> InferenceConfig:
        """Extract params set for inference from the internally saved configuration.
        Returns:
            InferenceConfig: extracted inference configuration
        Raises:
            KeyError: if required keys are missing in the configuration
            RuntimeError: if extraction fails for other reasons
        """
        try:
            params_set: Dict[str, Any] = {}
            
            params_set['device'] = self.get("device").get("type")
            params_set['model_dir'] = self.get("paths").get("models").get("deepfake_detection").get("dir")
            params_set['model_name'] = self.get("paths").get("models").get("deepfake_detection").get("filename")
            params_set['auto_download_model'] = self.get("model").get("deepfake_detection").get("auto_download")
            params_set['hf_model_url'] = self.get("model").get("deepfake_detection").get("huggingface").get("url")
            params_set['model_config'] = self._extract_model_config()
            params_set['face_detection_config'] = self._extract_face_detection_config()
            params_set['face_extraction_config'] = self._extract_face_extraction_config()
            params_set['imagenet_mean'] = self.get("preprocessing").get("image").get("normalization").get("mean")
            params_set['imagenet_std'] = self.get("preprocessing").get("image").get("normalization").get("std")
            
            return InferenceConfig(**params_set)
            
        except KeyError as e:
            err_mess = f"Missing required configuration key: {e}"
            self.logger.fatal(err_mess)
            raise KeyError(err_mess) from e
        
        except Exception as e:
            err_mess = f"Error extracting inference configuration: {e}"
            self.logger.fatal(err_mess)
            raise RuntimeError(err_mess) from e
    
    def _extract_model_config(self) -> ModelArchitectureConfig:
        """Extract model architecture configuration from the internally saved configuration.
        Returns:
            ModelArchitectureConfig: extracted model architecture configuration
        Raises:
            KeyError: if required keys are missing in the configuration
            RuntimeError: if extraction fails for other reasons
        """
        try:
            params_set: Dict[str, Any] = {}
            
            params_set['spatial_feature_config'] = self._extract_spatial_feature_config()
            params_set['frequency_feature_config'] = self._extract_frequency_feature_config()
            params_set['classifier_config'] = self._extract_classifier_config()
            params_set['concat_spatial_features_dim'] = self.get("model_architecture").get("concatenation").get("spatial_dim")
            params_set['concat_frequency_features_dim'] = self.get("model_architecture").get("concatenation").get("frequency_dim")
            params_set['score_aggregation_method'] = self.get("inference").get("score_aggregation").get("aggregation_method")
            
            return ModelArchitectureConfig(**params_set)
        
        except KeyError as e:
            err_mess = f"Missing required configuration key for model architecture: {e}"
            self.logger.fatal(err_mess)
            raise KeyError(err_mess) from e
        
        except Exception as e:
            err_mess = f"Error extracting model architecture configuration: {e}"
            self.logger.fatal(err_mess)
            raise RuntimeError(err_mess) from e
    
    def _extract_face_detection_config(self) -> FaceDetectionConfig:
        """Extract face detection configuration from the internally saved configuration.
        Returns:
            FaceDetectionConfig: extracted face detection configuration
            None: if face detection is disabled
        Raises:
            KeyError: if required keys are missing in the configuration
            RuntimeError: if extraction fails for other reasons
        """ 
        try:
            params_set: Dict[str, Any] = {}
            
            if self.get("preprocessing").get("face_detection").get("enabled"):
                params_set['model_path'] = self.get("paths").get("face_detection").get("custom_weights_path")
                params_set['min_face_size'] = self.get("preprocessing").get("face_detection").get("min_face_size")
                params_set['confidence_threshold'] = self.get("preprocessing").get("face_detection").get("confidence_threshold")
                params_set['only_keep_top_face'] = self.get("preprocessing").get("face_detection").get("only_keep_top_face")
            
                return FaceDetectionConfig(**params_set)
            
            return None
        
        except KeyError as e:
            err_mess = f"Missing required configuration key for face detection: {e}"
            self.logger.fatal(err_mess)
            raise KeyError(err_mess) from e
        
        except Exception as e:
            err_mess = f"Error extracting face detection configuration: {e}"
            self.logger.fatal(err_mess)
            raise RuntimeError(err_mess) from e
        
    def _extract_face_extraction_config(self) -> FaceExtractionConfig:
        """Extract face extraction configuration from the internally saved configuration.
        Returns:
            FaceExtractionConfig: extracted face extraction configuration
        Raises:
            KeyError: if required keys are missing in the configuration
            RuntimeError: if extraction fails for other reasons
        """
        try:
            params_set: Dict[str, Any] = {}
            
            params_set['target_size'] = tuple(self.get("preprocessing").get("face_extraction").get("target_size"))
            params_set['normalization'] = self.get("preprocessing").get("face_extraction").get("normalization")
            params_set['padding'] = self.get("preprocessing").get("face_extraction").get("padding")
            
            return FaceExtractionConfig(**params_set)
        
        except KeyError as e:
            err_mess = f"Missing required configuration key for face extraction: {e}"
            self.logger.fatal(err_mess)
            raise KeyError(err_mess) from e
        
        except Exception as e:
            err_mess = f"Error extracting face extraction configuration: {e}"
            self.logger.fatal(err_mess)
            raise RuntimeError(err_mess) from e
            
    def _extract_classifier_config(self) -> ClassifierConfig:
        """Extract classifier configuration from the internally saved configuration.
        Returns:
            ClassifierConfig: extracted classifier configuration
        Raises:
            KeyError: if required keys are missing in the configuration
            RuntimeError: if extraction fails for other reasons
        """
        try:
            params_set: Dict[str, Any] = {}
            
            params_set['input_dim'] = self.get("model_architecture").get("classifier").get("input_dim")
            params_set['num_classes'] = self.get("model_architecture").get("classifier").get("num_classes")
            params_set['hidden_dims'] = self.get("model_architecture").get("classifier").get("hidden_dims")
            params_set['dropout_rate'] = self.get("model_architecture").get("classifier").get("dropout_rate")
            params_set['use_batch_norm'] = self.get("model_architecture").get("classifier").get("use_batch_norm")
            params_set['path_to_weights'] = self.get("paths").get("classifier").get("custom_weights_path")
            
            return ClassifierConfig(**params_set)
        
        except KeyError as e:
            err_mess = f"Missing required configuration key for classifier: {e}"
            self.logger.fatal(err_mess)
            raise KeyError(err_mess) from e
        
        except Exception as e:
            err_mess = f"Error extracting classifier configuration: {e}"
            self.logger.fatal(err_mess)
            raise RuntimeError(err_mess) from e
    
    def _extract_spatial_feature_config(self) -> SpatialFeatureConfig:
        """Extract spatial feature extractor configuration from the internally saved configuration.
        Returns:
            SpatialFeatureConfig: extracted spatial feature extractor configuration
        Raises:
            KeyError: if required keys are missing in the configuration
            RuntimeError: if extraction fails for other reasons
        """
        try:
            params_set: Dict[str, Any] = {}
            
            params_set['effnet_pretrained'] = self.get("model_architecture").get("spatial_features").get("pretrained")
            params_set['path_to_weights'] = self.get("paths").get("efficientnet").get("custom_weights_path")
            
            return SpatialFeatureConfig(**params_set)
        
        except KeyError as e:
            err_mess = f"Missing required configuration key for spatial feature extractor: {e}"
            self.logger.fatal(err_mess)
            raise KeyError(err_mess) from e
        
        except Exception as e:
            err_mess = f"Error extracting spatial feature extractor configuration: {e}"
            self.logger.fatal(err_mess)
            raise RuntimeError(err_mess) from e
    
    def _extract_frequency_feature_config(self) -> FrequencyFeatureConfig:
        """Extract frequency feature extractor configuration from the internally saved configuration.
        Returns:
            FrequencyFeatureConfig: extracted frequency feature extractor configuration
        Raises:
            KeyError: if required keys are missing in the configuration
            RuntimeError: if extraction fails for other reasons
        """
        try:
            params_set: Dict[str, Any] = {}
            
            params_set['fft_feature_dim'] = self.get("model_architecture").get("frequency_features").get("fft").get("feature_dim")
            params_set['fft_num_radial_bands'] = self.get("model_architecture").get("frequency_features").get("fft").get("num_radial_bands")
            params_set['fft_window_function'] = self.get("model_architecture").get("frequency_features").get("fft").get("window_function")
            params_set['fft_high_freq_emphasis'] = self.get("model_architecture").get("frequency_features").get("fft").get("high_freq_emphasis")
            params_set['dct_feature_dim'] = self.get("model_architecture").get("frequency_features").get("dct").get("feature_dim")
            params_set['dct_block_size'] = self.get("model_architecture").get("frequency_features").get("dct").get("block_size")
            params_set['channel_mode'] = self.get("model_architecture").get("channel_mode")
            params_set['dct_aggregation_method'] = self.get("model_architecture").get("frequency_features").get("dct").get("aggregation_method")
            params_set['dct_num_frequency_bands'] = self.get("model_architecture").get("frequency_features").get("dct").get("num_frequency_bands")
            params_set['fusion_output_dim'] = self.get("model_architecture").get("frequency_features").get("fusion").get("output_dim")
            params_set['fusion_hidden_dims'] = self.get("model_architecture").get("frequency_features").get("fusion").get("hidden_dims")
            params_set['fusion_dropout_rate'] = self.get("model_architecture").get("frequency_features").get("fusion").get("dropout_rate")
            params_set['fusion_use_batch_norm'] = self.get("model_architecture").get("frequency_features").get("fusion").get("use_batch_norm")
            params_set['path_to_weights'] = self.get("paths").get("frequency_extractor").get("custom_weights_path")
            params_set['use_gpu_extractors'] = self.get("model_architecture").get("frequency_features").get("use_gpu_extractors")
            params_set['fft_num_azimuthal_sectors'] = self.get("model_architecture").get("frequency_features").get("fft").get("constants").get("num_azimuthal_sectors")
            params_set['fft_high_freq_threshold_ratio'] = self.get("model_architecture").get("frequency_features").get("fft").get("constants").get("high_freq_threshold_ratio")
            params_set['fft_artifact_edge_width'] = self.get("model_architecture").get("frequency_features").get("fft").get("constants").get("artifact_edge_width")
            params_set['fft_artifact_num_radial_samples'] = self.get("model_architecture").get("frequency_features").get("fft").get("constants").get("artifact_num_radial_samples")
            params_set['fft_artifact_center_region_size'] = self.get("model_architecture").get("frequency_features").get("fft").get("constants").get("artifact_center_region_size")
            params_set['fft_energy_band_start_multiplier'] = self.get("model_architecture").get("frequency_features").get("fft").get("constants").get("energy_band_start_multiplier")
            params_set['fft_energy_band_end_multiplier'] = self.get("model_architecture").get("frequency_features").get("fft").get("constants").get("energy_band_end_multiplier")
            params_set['fft_epsilon'] = self.get("model_architecture").get("frequency_features").get("fft").get("epsilon")
            params_set['dct_epsilon'] = self.get("model_architecture").get("frequency_features").get("dct").get("epsilon")
            params_set['fft_default_smoothing_kernel_size'] = self.get("model_architecture").get("frequency_features").get("fft").get("constants").get("default_smoothing_kernel_size")
            
            return FrequencyFeatureConfig(**params_set)
        
        except KeyError as e:
            err_mess = f"Missing required configuration key for frequency feature extractor: {e}"
            self.logger.fatal(err_mess)
            raise KeyError(err_mess) from e
        
        except Exception as e:
            err_mess = f"Error extracting frequency feature extractor configuration: {e}"
            self.logger.fatal(err_mess)
            raise RuntimeError(err_mess) from e
    
    def extract_config_type(self) -> str:
        """Determine the type of the internally saved configuration (e.g., 'inference', 'training').
        Returns:
            str: type of the configuration
        Raises:
            KeyError: if configuration type cannot be determined
        """
        type: str = self.get('type', None)
        if not type:
            err_mess = "Configuration type not specified."
            self.logger.fatal(err_mess)
            raise KeyError(err_mess)
        return type.lower()
    
    def extract_logger_config(self) -> Dict[str, Any]:
        """Retrieve logging configuration from the internally saved configuration.
        Returns:
            Dict[str, Any]: logging configuration dictionary
        Raises:
            KeyError: if logging configuration is missing
        """
        try:
            params_set: Dict[str, Any] = {}
            params_set['log_dir'] = self.get('paths').get('output').get('logs_dir')
            params_set['log_filename'] = params_set['log_dir'] + '/' + self.get('logging').get('file').get('path')
            params_set['log_level'] = self.get('logging').get('level')
            params_set['log_format'] = self.get('logging').get('format')
            params_set['log_datefmt'] = self.get('logging').get('datefmt')
            params_set['log_console_enabled'] = self.get('logging').get('console').get('enabled')
            params_set['log_file_enabled'] = self.get('logging').get('file').get('enabled')
            
            return params_set
            
        except KeyError as e:
            err_mess = f"Missing required configuration key for logging: {e}"
            raise KeyError(err_mess) from e
        
    def extract_training_config(self) -> TrainingConfig:
        """Extract training configuration from the internally saved configuration.
        Returns:
            TrainingConfig: extracted training configuration
        Raises:
            KeyError: if required keys are missing in the configuration
            RuntimeError: if extraction fails for other reasons
        """
        try:
            params_set: Dict[str, Any] = {}
            
            params_set["device"] = self.get("device").get("type")
            params_set["train_csv_path"] = self.get("paths").get("datasets").get("training").get("csv_file")
            params_set["test_csv_path"] = self.get("paths").get("datasets").get("testing").get("csv_file")
            params_set["eval_csv_path"] = self.get("paths").get("datasets").get("validation").get("csv_file")
            params_set["dataloaders_batch_size"] = self.get("training").get("dataloaders").get("batch_size")
            params_set["num_workers"] = self.get("training").get("dataloaders").get("num_workers")
            params_set["learning_rate"] = self.get("training").get("optimizer").get("learning_rate")
            params_set["min_learning_rate"] = self.get("training").get("optimizer").get("min_learning_rate")
            params_set["weight_decay"] = self.get("training").get("optimizer").get("weight_decay")
            params_set["num_epochs"] = self.get("training").get("num_epochs")
            params_set["warmup_epochs"] = self.get("training").get("warmup_epochs")
            params_set["compile_model"] = self.get("training").get("compile_model")
            params_set["seed"] = self.get("training").get("seed")
            params_set["use_automixed_precision"] = self.get("training").get("use_automixed_precision")
            params_set["log_interval"] = self.get("training").get("logging").get("log_interval")
            params_set["log_gradients"] = self.get("training").get("logging").get("log_gradients")
            params_set["log_weights"] = self.get("training").get("logging").get("log_weights")
            params_set["save_dir"] = self.get("paths").get("output").get("model_checkpoints_dir")
            params_set["eval_interval"] = self.get("training").get("evaluation").get("eval_interval")
            params_set["early_stopping_patience"] = self.get("training").get("evaluation").get("early_stopping_patience")
            params_set["wandb_project"] = self.get("logging").get("wandb").get("project")
            params_set["wandb_entity"] = self.get("logging").get("wandb").get("entity")
            params_set["use_wandb"] = self.get("logging").get("wandb").get("enabled")
            
            return TrainingConfig(**params_set)
        
        except KeyError as e:
            err_mess = f"Missing required configuration key for training: {e}"
            self.logger.fatal(err_mess)
            raise KeyError(err_mess) from e
        
        except Exception as e:
            err_mess = f"Error extracting training configuration: {e}"
            self.logger.fatal(err_mess)
            raise RuntimeError(err_mess) from e
        
    def extract_predict_video_params(self) -> Dict[str, Any]:
        """Extract parameters for video prediction from the internally saved configuration.
        Returns:
            Dict[str, Any]: extracted video prediction parameters
        Raises:
            KeyError: if required keys are missing in the configuration
            RuntimeError: if extraction fails for other reasons
        """
        try:
            params_set: Dict[str, Any] = {}
            
            params_set["aggregation_method"]=self.get('inference').get('score_aggregation').get('aggregation_method')
            params_set["batch_size"]=self.get('inference').get('batch_processing').get('default_batch_size')
            params_set["frames_per_second"]=self.get('preprocessing').get('frame_extraction').get('frames_per_second')
            params_set["use_pyav_extractor"]=self.get('preprocessing').get('frame_extraction').get('use_optimized') 
            params_set["enable_gradcam"]=self.get('inference').get('gradcam').get('enabled')
            params_set["gradcam_output_path"]=self.get('paths').get('output').get('gradcam_dir')
            params_set["gradcam_overlay_alpha"]=self.get('inference').get('gradcam').get('overlay_alpha')
            
            return params_set
        
        except KeyError as e:
            err_mess = f"Missing required configuration key for video prediction parameters: {e}"
            self.logger.fatal(err_mess)
            raise KeyError(err_mess) from e
        
        except Exception as e:
            err_mess = f"Error extracting video prediction parameters: {e}"
            self.logger.fatal(err_mess)
            raise RuntimeError(err_mess) from e
        
    def extract_preprocessing_config(self) -> PreprocessingConfig:
        """Extract preprocessing configuration from the internally saved configuration.
        Returns:
            PreprocessingConfig: extracted preprocessing configuration
        Raises:
            KeyError: if required keys are missing in the configuration
            RuntimeError: if extraction fails for other reasons
        """
        try:
            params_set: Dict[str, Any] = {}
            
            params_set['frames_per_second'] = self.get("preprocessing").get("frame_extraction").get("frames_per_second")
            params_set['min_face_size'] = self.get("preprocessing").get("face_detection").get("min_face_size")
            params_set['confidence_threshold'] = self.get("preprocessing").get("face_detection").get("confidence_threshold")
            params_set['only_keep_top_face'] = self.get("preprocessing").get("face_detection").get("only_keep_top_face")
            params_set['face_target_size'] = tuple(self.get("preprocessing").get("face_extraction").get("target_size"))
            params_set['face_normalization'] = self.get("preprocessing").get("face_extraction").get("normalization")
            params_set['face_padding'] = self.get("preprocessing").get("face_extraction").get("padding")
            
            return PreprocessingConfig(**params_set)
        
        except KeyError as e:
            err_mess = f"Missing required configuration key for preprocessing: {e}"
            self.logger.fatal(err_mess)
            raise KeyError(err_mess) from e
        
        except Exception as e:
            err_mess = f"Error extracting preprocessing configuration: {e}"
            self.logger.fatal(err_mess)
            raise RuntimeError(err_mess) from e
        
    def extract_dataloader_config(self) -> DataLoaderConfig:
        """Extract dataloader configuration from the internally saved configuration.
        Returns:
            DataLoaderConfig: extracted dataloader configuration
        Raises:
            KeyError: if required keys are missing in the configuration
            RuntimeError: if extraction fails for other reasons
        """
        try:
            params_set: Dict[str, Any] = {}
            
            params_set['num_workers'] = self.get("training").get("dataloaders").get("num_workers")
            params_set['device'] = self.get("device").get("type")
            params_set['preprocessing_config'] = self.extract_preprocessing_config()
            params_set['csv_folder'] = self.get("paths").get("datasets").get("csv_folder")
            params_set['data_folder'] = self.get("paths").get("datasets").get("raw_data_folder")
            params_set['frames_dir'] = self.get("paths").get("datasets").get("frames_dir")
            params_set['faces_dir'] = self.get("paths").get("datasets").get("faces_dir")
            
            return DataLoaderConfig(**params_set)
        
        except KeyError as e:
            err_mess = f"Missing required configuration key for dataloader: {e}"
            self.logger.fatal(err_mess)
            raise KeyError(err_mess) from e
        
        except Exception as e:
            err_mess = f"Error extracting dataloader configuration: {e}"
            self.logger.fatal(err_mess)
            raise RuntimeError(err_mess) from e
        
    def extract_build_dataset_params(self) -> Dict[str, Any]:
        """Extract parameters for building the dataset from the internally saved configuration.
        Returns:
            Dict[str, Any]: extracted dataset building parameters
        Raises:
            KeyError: if required keys are missing in the configuration
            RuntimeError: if extraction fails for other reasons
        """
        try:
            params_set: Dict[str, Any] = {}
            
            params_set["name"] = self.get("dataset").get("name")
            params_set["balance"] = self.get("dataset").get("balance").get("enabled")
            params_set["balance_method"] = self.get("dataset").get("balance").get("method")
            params_set["split"] = self.get("dataset").get("split").get("enabled")
            params_set["train_ratio"] = self.get("dataset").get("split").get("train_ratio")
            params_set["test_ratio"] = self.get("dataset").get("split").get("test_ratio")
            params_set["eval_ratio"] = self.get("dataset").get("split").get("eval_ratio")
            params_set["save_preprocessed"] = self.get("dataset").get("save_preprocessed")
            params_set["random_state"] = self.get("dataset").get("random_state")
            
            return params_set
        
        except KeyError as e:
            err_mess = f"Missing required configuration key for dataset building parameters: {e}"
            self.logger.fatal(err_mess)
            raise KeyError(err_mess) from e
        
        except Exception as e:
            err_mess = f"Error extracting dataset building parameters: {e}"
            self.logger.fatal(err_mess)
            raise RuntimeError(err_mess) from e