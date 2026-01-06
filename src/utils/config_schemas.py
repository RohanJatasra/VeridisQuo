"""Dataclasses defining configuration schemas for the DeepFakeDetector model architecture and training.
These classes are not meant to be instantiated directly, not read by humans. They serve as structured containers"""

from dataclasses import dataclass
from torch import device
from typing import Optional, Any, Tuple


@dataclass
class SpatialFeatureConfig:
    """Dataclass storing spatial feature extractor parameters."""
    effnet_pretrained: bool
    """Whether to use pretrained weights for EfficientNet-B4."""
    path_to_weights: Optional[str] =None
    """Path to pretrained weights for EfficientNet-B4."""
    
@dataclass
class FrequencyFeatureConfig:
    """Dataclass storing frequency feature extractor parameters."""
    fft_feature_dim: int
    """Dimensionality of FFT features."""
    fft_num_radial_bands: int
    """Number of radial frequency bands for FFT features."""
    fft_window_function: str
    """Window function to apply before FFT computation."""
    fft_high_freq_emphasis: bool
    """Whether to apply high-frequency emphasis to FFT features."""
    dct_feature_dim: int
    """Dimensionality of DCT features."""
    dct_block_size: int
    """Block size for DCT computation."""
    channel_mode: str
    """Channel mode for frequency feature extraction (e.g., 'rgb', 'ycbcr')."""
    dct_aggregation_method: str
    """Method to aggregate DCT coefficients."""
    dct_num_frequency_bands: int
    """Number of frequency bands for DCT features."""
    fusion_output_dim: int
    """Output dimensionality of the frequency feature fusion module."""
    fusion_hidden_dims: Optional[list[int]]
    """Hidden layer dimensions for the fusion MLP."""
    fusion_dropout_rate: float
    """Dropout rate for the fusion MLP."""
    fusion_use_batch_norm: bool
    """Whether to use batch normalization in the fusion MLP."""
    path_to_weights: Optional[str]
    """Path to pretrained weights for the frequency feature extractor fusion module."""
    use_gpu_extractors: bool
    """Whether to use GPU for frequency feature extraction."""
    fft_num_azimuthal_sectors: int
    """Number of azimuthal sectors for FFT artifact mitigation."""
    fft_high_freq_threshold_ratio: float
    """Threshold ratio for high-frequency artifact mitigation."""
    fft_artifact_edge_width: int
    """Edge width for FFT artifact mitigation."""
    fft_artifact_num_radial_samples: int
    """Number of radial samples for FFT artifact mitigation."""
    fft_artifact_center_region_size: int
    """Center region size for FFT artifact mitigation."""
    fft_energy_band_start_multiplier: float
    """Start multiplier for FFT energy band selection."""
    fft_energy_band_end_multiplier: float
    """End multiplier for FFT energy band selection."""
    fft_epsilon: float
    """Small constant to avoid division by zero in FFT computations."""
    dct_epsilon: float
    """Small constant to avoid division by zero in DCT computations."""
    fft_default_smoothing_kernel_size: int
    """Default smoothing kernel size for FFT features."""

@dataclass
class ClassifierConfig:
    """Dataclass storing classifier parameters."""
    input_dim: int
    """Input feature dimension for the classifier."""
    num_classes: int
    """Number of output classes (e.g., 2 for REAL/FAKE)."""
    hidden_dims: Optional[list[int]]
    """Hidden layer dimensions for the classifier MLP."""
    dropout_rate: float
    """Dropout rate for regularization in the classifier."""
    use_batch_norm: bool
    """Whether to use batch normalization in the classifier."""
    path_to_weights: Optional[str] =None
    """Path to pretrained weights for the classifier."""

@dataclass
class ModelArchitectureConfig:
    """Dataclass storing deepfake detection model architecture parameters."""
    spatial_features: SpatialFeatureConfig
    """Stores spatial feature extractor parameters."""
    
    frequency_features: FrequencyFeatureConfig
    """Stores frequency feature extractor parameters."""
    
    classifier: ClassifierConfig
    """Stores classifier parameters."""
    
    concat_spatial_features_dim: int
    """Dimensionality of spatial features to be concatenated."""
    concat_frequency_features_dim: int
    """Dimensionality of frequency features to be concatenated."""
    
    score_aggregation_method: str
    """Method for aggregating frame-level scores."""

@dataclass
class FaceDetectionConfig:
    """Dataclass storing face detection parameters."""
    model_path: str
    """Path to the face detection model."""
    
    min_face_size: int
    """Minimum face size to detect."""
    confidence_threshold: float
    """Confidence threshold for face detection."""
    only_keep_top_face: bool
    """Whether to keep only the top detected face."""
    
@dataclass
class FaceExtractionConfig:
    """Dataclass storing face extraction parameters."""
    target_size: tuple[int, int]
    """Target size (height, width) for extracted face images."""
    normalization: dict
    """Normalization parameters (mean, std) for face images."""
    padding: int
    """Padding around detected face bounding box."""

@dataclass
class InferenceConfig:
    """Dataclass storing inference configuration parameters."""
    device: device
    """Device for inference (e.g., 'cpu' or 'cuda')."""

    model_dir: str
    """Directory path where the model weights are stored."""
    model_name: str
    """Filename of the model weights."""
    model_config: ModelArchitectureConfig
    """Stores deepfake detection model parameters, for each submodule."""
    face_detection_config: FaceDetectionConfig
    """Stores face detection model parameters."""
    face_extraction_config: FaceExtractionConfig
    """Stores face extraction parameters."""
    imagenet_mean: list[float]
    """ImageNet mean for input normalization."""
    imagenet_std: list[float]
    """ImageNet standard deviation for input normalization."""
    auto_download_model: Optional[bool] =True
    """Whether to automatically download the model if not found locally."""
    hf_model_url: Optional[str] =None
    """Hugging Face URL to download the model from."""
    
@dataclass
class TrainingConfig:
    """Configuration dataclass for DeepFakeDetector model training.
    Contains all hyperparameters, paths, and settings required for training loop execution.
    """
    device: device
    """Device for training (e.g., 'cpu' or 'cuda')."""

    train_csv_path: str
    """Path to training dataset CSV file."""
    test_csv_path: str
    """Path to test dataset CSV file."""
    eval_csv_path: str
    """Path to validation dataset CSV file."""

    dataloaders_batch_size: int
    """Batch size for dataloaders."""
    num_workers: int
    """Number of worker threads for data loading."""

    learning_rate: float
    """Initial learning rate for AdamW optimizer."""
    min_learning_rate: float
    """Minimum learning rate after cosine annealing."""
    weight_decay: float
    """L2 regularization weight decay factor."""
    num_epochs: int
    """Total number of training epochs."""
    warmup_epochs: int
    """Number of linear warmup epochs before cosine annealing."""

    compile_model: bool
    """Whether to use torch.compile for model optimization."""
    seed: int
    """Random seed for reproducibility."""
    use_automixed_precision: bool
    """Whether to use automatic mixed precision training."""

    log_interval: int
    """Batch interval for logging training metrics."""
    log_gradients: bool
    """Whether to log gradients to wandb."""
    log_weights: bool
    """Whether to log model weights to wandb."""
    save_dir: str
    """Directory path for saving model checkpoints."""

    eval_interval: int
    """Epoch interval for validation evaluation."""
    early_stopping_patience: int
    """Number of epochs without improvement before stopping."""

    wandb_project: str
    """Weights & Biases project name."""
    use_wandb: bool
    """Whether to enable Weights & Biases logging."""
    wandb_entity: Optional[str] =None
    """Weights & Biases entity name (team/user)."""
   
@dataclass
class PreprocessingConfig:
    """Dataclass storing preprocessing parameters."""
    frames_per_second: int
    """Frame extraction rate for video preprocessing."""
    min_face_size: int
    """Minimum face size to detect."""
    confidence_threshold: float
    """Confidence threshold for face detection."""
    only_keep_top_face: bool
    """Whether to keep only the top detected face."""
    face_target_size: Tuple[int, int]
    """Target size for face images."""
    face_normalization: str 
    """Normalization method for face images."""
    face_padding: int
    """Padding around detected faces."""
    
@dataclass
class DataLoaderConfig:
    """Dataclass storing dataloader parameters."""
    device: str
    """Device for data loading (e.g., 'cpu' or 'cuda')."""
    preprocessing_config: PreprocessingConfig
    csv_folder: str
    """Path to folder containing CSV files."""
    data_folder: str
    """Path to folder containing raw data."""
    frames_dir: str
    """Path to folder for storing extracted frames."""
    faces_dir: str
    """Path to folder for storing extracted face images."""
    num_workers: Optional[int] =None
    """Number of worker threads for data loading."""
 
    