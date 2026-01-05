from torch import nn
import torch
from spatial import EfficientNetFeatureExtractor
from frequal import FrequencyFeatureExtractor
from classification import FaceFeaturesConcatenator, FaceClassifier
from inference import ScoreAggregator, AggregatedScore
from logging import Logger, getLogger
from core import ModelArchitectureConfig

class DeepFakeDetector(nn.Module):
    """Deepfake detection model combining spatial and frequency analysis."""

    def __init__(self, config: ModelArchitectureConfig) -> None:
        """Initialize DeepFakeDetector.
        Parameters:
            effnet_pretrained (bool): whether to use effnet_pretrained weights for EfficientNet-B4 (legacy, overridden by config)
            use_gpu (bool): whether to use GPU for frequency feature extraction (legacy, overridden by config)
            config (ModelArchitectureConfig, optional): model architecture configuration
        Raises:
            RuntimeError: if GPU requested but CUDA is not available
        """
        super(DeepFakeDetector, self).__init__()

        self.logger: Logger = getLogger("/".join(__file__.split("/")[-2:]))
        """Logger instance for the DeepFakeDetector class"""

        if config is None:
            error_msg = "ModelArchitectureConfig is required to initialize DeepFakeDetector."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.config: ModelArchitectureConfig = config
        """Model architecture configuration"""

        self.efficient: EfficientNetFeatureExtractor = EfficientNetFeatureExtractor(
            effnet_pretrained=self.config.spatial_features.effnet_pretrained,
            path_to_weights=self.config.spatial_features.path_to_weights
        )
        """EfficientNet-B4 feature extractor for spatial analysis"""

        self.frequal_extractor: FrequencyFeatureExtractor = FrequencyFeatureExtractor(
            fft_feature_dim=self.config.frequency_features.fft_feature_dim,
            fft_num_radial_bands=self.config.frequency_features.fft_num_radial_bands,
            fft_window_function=self.config.frequency_features.fft_window_function,
            fft_high_freq_emphasis=self.config.frequency_features.fft_high_freq_emphasis,
            dct_feature_dim=self.config.frequency_features.dct_feature_dim,
            dct_block_size=self.config.frequency_features.dct_block_size,
            channel_mode=self.config.frequency_features.channel_mode,
            dct_aggregation_method=self.config.frequency_features.dct_aggregation_method,
            dct_num_frequency_bands=self.config.frequency_features.dct_num_frequency_bands,
            fusion_output_dim=self.config.frequency_features.fusion_output_dim,
            fusion_hidden_dims=self.config.frequency_features.fusion_hidden_dims,
            fusion_dropout_rate=self.config.frequency_features.fusion_dropout_rate,
            fusion_use_batch_norm=self.config.frequency_features.fusion_use_batch_norm,
            path_to_weights=self.config.frequency_features.path_to_weights,
            use_gpu_extractors=self.config.frequency_features.use_gpu_extractors,
            fft_num_azimuthal_sectors=self.config.frequency_features.fft_num_azimuthal_sectors,
            fft_high_freq_threshold_ratio=self.config.frequency_features.fft_high_freq_threshold_ratio,
            fft_artifact_edge_width=self.config.frequency_features.fft_artifact_edge_width,
            fft_artifact_num_radial_samples=self.config.frequency_features.fft_artifact_num_radial_samples,
            fft_artifact_center_region_size=self.config.frequency_features.fft_artifact_center_region_size,
            fft_default_smoothing_kernel_size=self.config.frequency_features.fft_default_smoothing_kernel_size,
            fft_energy_band_start_multiplier=self.config.frequency_features.fft_energy_band_start_multiplier,
            fft_energy_band_end_multiplier=self.config.frequency_features.fft_energy_band_end_multiplier,
            fft_epsilon=self.config.frequency_features.fft_epsilon,
            dct_epsilon=self.config.frequency_features.dct_epsilon
        )
        """Frequency feature extractor using FFT and DCT"""

        self.concatenator: FaceFeaturesConcatenator = FaceFeaturesConcatenator(
            spatial_feature_dim=self.config.concat_spatial_features_dim,
            frequency_feature_dim=self.config.concat_frequency_features_dim
        )
        """Concatenates spatial and frequency features"""

        self.classifier: FaceClassifier = FaceClassifier(
            input_dim=config.classifier.input_dim,
            num_classes=config.classifier.num_classes,
            hidden_dims=config.classifier.hidden_dims,
            dropout_rate=config.classifier.dropout_rate,
            use_batch_norm=config.classifier.use_batch_norm,
            path_to_weights=config.classifier.path_to_weights
        )
        """Face classifier taking concatenated features and outputting logits"""

        self.aggregator: ScoreAggregator = ScoreAggregator(self.config.score_aggregation_method)
        """ScoreAggregator for aggregating frame-level predictions"""

    def forward(self, x_image: torch.Tensor) -> torch.Tensor:
        """Forward pass through the deepfake detection model.
        Parameters:
            x_image (torch.Tensor): Input tensor [batch, 3, 224, 224]
                - Format: RGB (not BGR)
                - Normalized with ImageNet statistics
                - Values typically in range [-2.5, 2.5] after normalization
        Returns:
            torch.Tensor: Logits [batch, 2] for [REAL, FAKE] classes
                - Raw scores (not probabilities)
                - Apply softmax for probabilities: torch.softmax(output, dim=1)
                - Apply argmax for predictions: torch.argmax(output, dim=1)
        Raises:
            AssertionError: if input shape is not (batch, 3, 224, 224)
        """
        try:
            assert x_image.shape[1:] == (3, 224, 224), f"Input must be [batch, 3, 224, 224], got {x_image.shape}"
        
        except AssertionError as e:
            err_msg = f"Invalid input shape: {e}"
            self.logger.error(err_msg)
            raise AssertionError(err_msg) from e

        # Extract spatial features from EfficientNet-B4
        # Returns: (intermediate_features, pooled_features [B, 1792])
        _, spatial_features = self.efficient(x_image)

        # Extract frequency features from FFT + DCT
        # Returns: fused_features [B, 1024]
        frequency_features = self.frequal_extractor(x_image)

        # Concatenate spatial and frequency features
        # [B, 1792] + [B, 1024] → [B, 2816]
        combined_features = self.concatenator(spatial_features, frequency_features)

        # Classification: [B, 2816] → [B, 2]
        output = self.classifier(combined_features)

        return output