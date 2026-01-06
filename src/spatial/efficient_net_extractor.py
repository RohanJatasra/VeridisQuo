import torch
import torch.nn as nn
from torchvision.models import EfficientNet, efficientnet_b4
from typing import Optional, Tuple

class EfficientNetFeatureExtractor(nn.Module):
    """EfficientNet-based feature extractor for spatial analysis.
    Uses EfficientNet-B4 architecture as backbone to extract both intermediate
    convolutional features and pooled global features for deepfake detection."""

    def __init__(self, path_to_weights: Optional[str], pretrained: bool) -> None:
        """Initialize EfficientNetFeatureExtractor.
        Parameters:
            path_to_weights (Optional[str]): path to custom pretrained weights file, or None to skip custom weight loading
            pretrained (bool): whether to use ImageNet pretrained weights for the backbone
        """
        super(EfficientNetFeatureExtractor, self).__init__()

        #load EfficientNet-B4 backbone with optional ImageNet pretraining
        weights = 'DEFAULT' if pretrained else None
        self.backbone: EfficientNet = efficientnet_b4(weights=weights)
        """EfficientNet-B4 backbone model for feature extraction."""

        #load custom weights if provided
        if path_to_weights is not None:
            self.load_weights(path_to_weights)

        self.features = self.backbone.features
        """Sequential container of convolutional feature extraction layers from the backbone."""

        self.avgpool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        """Adaptive average pooling layer to produce fixed-size global features."""
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract spatial features from input images.
        Parameters:
            x (torch.Tensor): input batch of images with shape [batch, channels, height, width]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple containing:
                - features (torch.Tensor): convolutional feature maps with shape [batch, feature_channels, h, w]
                - pooled_features (torch.Tensor): globally pooled features with shape [batch, feature_channels, 1, 1]
        """
        features = self.features(x)
        pooled_features = self.avgpool(features)
        return features, pooled_features

    def save_weights(self, path: str) -> None:
        """Save the backbone model weights to disk.
        Parameters:
            path (str): file path where weights will be saved
        """
        torch.save(self.backbone.state_dict(), path)

    def load_weights(self, path: str) -> None:
        """Load pretrained weights into the backbone model.
        Parameters:
            path (str): file path to the saved weights
        """
        state_dict = torch.load(path)
        self.backbone.load_state_dict(state_dict)
        

