import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepfake-detector'))

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
        
    def test_save_weights(self, tmp_path):
        model = EfficientNetFeatureExtractor(pretrained=False)
        
        save_path = tmp_path / "efficientnet_weights.pth"
        model.save_weights(str(save_path))
        
        assert save_path.exists(), "Weights file was not saved properly."
        
    def test_load_weights(self, tmp_path):
        model = EfficientNetFeatureExtractor(pretrained=False)
        
        save_path = tmp_path / "efficientnet_weights.pth"
        model.save_weights(str(save_path))
        
        # Create a new model instance and load weights
        new_model = EfficientNetFeatureExtractor(pretrained=False)
        new_model.load_weights(str(save_path))
        
        for param1, param2 in zip(model.backbone.parameters(), new_model.backbone.parameters()):
            assert torch.equal(param1, param2), "Weights were not loaded properly."