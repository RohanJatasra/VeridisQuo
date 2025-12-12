import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

from spatial.efficient_net import EfficientNetFeatureExtractor


class TestEfficientNetFeatureExtractor:

    def test_model_forward_single_image(self):
        model = EfficientNetFeatureExtractor(pretrained=False)
        
        sample_image_tensor = torch.randn(1, 3, 224, 224)
        features, pooled_features = model(sample_image_tensor)
        
        assert features.shape == (1, 1792, 7, 7)  , f"Expected features shape (1, 1792, 7, 7), got {features.shape}"
        assert pooled_features.shape == (1, 1792, 1, 1), f"Expected pooled_features shape (1, 1792, 1, 1), got {pooled_features.shape}"
        
    def test_batch_processing(self):
        model = EfficientNetFeatureExtractor(pretrained=False)
        
        sample_batch_tensor = torch.randn(4, 3, 224, 224)
        features, pooled_features = model(sample_batch_tensor)
        
        assert features.shape == (4, 1792, 7, 7), f"Expected features shape (4, 1792, 7, 7), got {features.shape}"
        assert pooled_features.shape == (4, 1792, 1, 1), f"Expected pooled_features shape (4, 1792, 1, 1), got {pooled_features.shape}"