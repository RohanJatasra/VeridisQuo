from torch import nn
import torch

from spatial.efficient_net import EfficientNetFeatureExtractor
from frequal.frequency_feature_extractor import FrequencyFeatureExtractor
from classification.face_features_concatenator import FaceFeaturesConcatenator
from classification.face_classifier import FaceClassifier


class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.efficient = EfficientNetFeatureExtractor(pretrained=True)
        self.frequal_extractor = FrequencyFeatureExtractor(use_gpu_extractors=True)
        
        self.concatenator = FaceFeaturesConcatenator()
        self.classifier = FaceClassifier(input_dim=2816)
        
        
        
    def forward(self, x_image : torch.Tensor):
        assert x_image.shape[1:] == (3,224,224), "Input image must have shape (3, 224, 224)"

        _, spatial_features = self.efficient(x_image)
        frequency_features = self.frequal_extractor(x_image)
        combined_features = self.concatenator(spatial_features, frequency_features)
        output = self.classifier(combined_features)

        return output