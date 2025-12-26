from typing import List
from torch import nn
import torch
from spatial.efficient_net import EfficientNetFeatureExtractor
from frequal.frequency_feature_extractor import FrequencyFeatureExtractor
from classification.face_features_concatenator import FaceFeaturesConcatenator
from classification.face_classifier import FaceClassifier
from inference import ScoreAggregator, AggregatedScore
from logging import Logger, getLogger

class DeepFakeDetector(nn.Module):
    """Deepfake detection model combining spatial and frequency analysis."""

    def __init__(self, pretrained: bool, use_gpu: bool) -> None:
        """Initialize DeepFakeDetector.
        Parameters:
            pretrained (bool): Whether to use pretrained weights for EfficientNet-B4.
            use_gpu (bool): Whether to use GPU for frequency feature extraction.
        Raises:
            RuntimeError: if GPU requested but CUDA is not available
        """
        super(DeepFakeDetector, self).__init__()

        self.logger: Logger = getLogger("/".join(__file__.split("/")[-2:]))
        """Logger instance for the DeepFakeDetector class"""
        
        self.efficient: EfficientNetFeatureExtractor = EfficientNetFeatureExtractor(
            pretrained=pretrained
        )
        """EfficientNet-B4 feature extractor for spatial analysis"""

        self.frequal_extractor: FrequencyFeatureExtractor = FrequencyFeatureExtractor(
            use_gpu_extractors=use_gpu
        )
        """Frequency feature extractor using FFT and DCT"""

        self.concatenator: FaceFeaturesConcatenator = FaceFeaturesConcatenator()
        """Concatenates spatial and frequency features"""
        
        self.classifier: FaceClassifier = FaceClassifier(input_dim=2816)
        """Face classifier taking concatenated features and outputting logits"""
        
        self.aggregator: ScoreAggregator = ScoreAggregator()
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