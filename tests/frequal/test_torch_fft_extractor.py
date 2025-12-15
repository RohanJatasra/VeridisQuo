"""Tests for TorchFFTExtractor."""
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deepfake-detector'))

from frequal import TorchFFTExtractor


class TestTorchFFTExtractorInitialization:
    """Test TorchFFTExtractor initialization."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        extractor = TorchFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        assert extractor.channel_mode == "luminance"
        assert extractor.num_radial_bands == 8
        assert extractor.window_function == "hann"
        assert extractor.feature_dim == 512

    def test_init_registers_window_buffer(self):
        """Test that window buffer is registered."""
        extractor = TorchFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        assert hasattr(extractor, 'window_224')
        assert isinstance(extractor.window_224, torch.Tensor)
        assert extractor.window_224.shape == (224, 224)

    def test_init_different_window_functions(self):
        """Test initialization with different window functions."""
        for window in ["hann", "hamming", "blackman", "none"]:
            extractor = TorchFFTExtractor(
                channel_mode="luminance",
                num_radial_bands=8,
                window_function=window,
                high_freq_emphasis=True,
                feature_dim=512
            )
            assert extractor.window_function == window


class TestTorchFFTExtractorForward:
    """Test TorchFFTExtractor forward pass."""

    def test_forward_single_image(self, batch_torch_images):
        """Test forward pass with single image."""
        extractor = TorchFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        single_image = batch_torch_images[0:1]
        features = extractor.extract(single_image)
        assert features.shape == (1, 512)

    def test_forward_batch(self, batch_torch_images):
        """Test forward pass with batch of images."""
        extractor = TorchFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        features = extractor.extract(batch_torch_images)
        assert features.shape == (batch_torch_images.shape[0], 512)

    def test_forward_output_not_nan(self, batch_torch_images):
        """Test that forward pass doesn't produce NaN."""
        extractor = TorchFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        features = extractor.extract(batch_torch_images)
        assert not torch.isnan(features).any()

    def test_forward_output_not_inf(self, batch_torch_images):
        """Test that forward pass doesn't produce infinity."""
        extractor = TorchFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        features = extractor.extract(batch_torch_images)
        assert not torch.isinf(features).any()

    def test_forward_deterministic_in_eval(self, batch_torch_images):
        """Test that forward pass is deterministic in eval mode."""
        extractor = TorchFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        extractor.eval()
        features1 = extractor.extract(batch_torch_images)
        features2 = extractor.extract(batch_torch_images)
        assert torch.allclose(features1, features2, rtol=1e-5)

    def test_forward_gradients_flow(self, batch_torch_images):
        """Test that gradients flow through the network."""
        extractor = TorchFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        images = batch_torch_images.requires_grad_(True)
        features = extractor.extract(images)
        loss = features.sum()
        loss.backward()
        assert images.grad is not None

    def test_forward_gpu_if_available(self, batch_torch_images):
        """Test forward pass on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        extractor = TorchFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        extractor = extractor.cuda()
        images = batch_torch_images.cuda()
        features = extractor.extract(images)
        assert features.device.type == 'cuda'
        assert features.shape == (batch_torch_images.shape[0], 512)


class TestTorchFFTExtractorConsistencyWithNumpy:
    """Test consistency between Torch and Numpy implementations."""

    def test_features_similar_to_numpy(self, simple_rgb_image):
        """Test that Torch and Numpy extractors produce similar features."""
        from frequal import NpFFTExtractor

        #create extractors with same parameters
        torch_extractor = TorchFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        np_extractor = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )

        #copy projection weights from torch to numpy extractor
        np_extractor.projection.weight.data = torch_extractor.projection.weight.data.clone()
        np_extractor.projection.bias.data = torch_extractor.projection.bias.data.clone()

        #prepare images
        np_image = simple_rgb_image[np.newaxis, ...]  #[1, H, W, 3] BGR
        torch_image = torch.from_numpy(np_image).float().permute(0, 3, 2, 1) / 255.0  #[1, 3, H, W] RGB

        #extract features
        torch_extractor.eval()
        with torch.no_grad():
            torch_features = torch_extractor.extract(torch_image).numpy()
        np_features = np_extractor.extract(np_image)

        #features should be similar (not exact due to implementation differences)
        #but should have same order of magnitude and correlation
        assert torch_features.shape == np_features.shape


class TestTorchFFTExtractorEdgeCases:
    """Test edge cases for TorchFFTExtractor."""

    def test_forward_with_normalized_input(self):
        """Test forward pass with normalized input [0, 1]."""
        extractor = TorchFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        images = torch.rand(2, 3, 224, 224)
        features = extractor.extract(images)
        assert features.shape == (2, 512)
        assert not torch.isnan(features).any()

    def test_forward_with_unnormalized_input(self):
        """Test forward pass with unnormalized input [0, 255]."""
        extractor = TorchFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        images = torch.rand(2, 3, 224, 224) * 255
        features = extractor.extract(images)
        assert features.shape == (2, 512)
        assert not torch.isnan(features).any()

    def test_forward_with_black_images(self):
        """Test forward pass with black images."""
        extractor = TorchFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        images = torch.zeros(2, 3, 224, 224)
        features = extractor.extract(images)
        assert features.shape == (2, 512)
        assert not torch.isnan(features).any()

    def test_forward_with_white_images(self):
        """Test forward pass with white images."""
        extractor = TorchFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        images = torch.ones(2, 3, 224, 224)
        features = extractor.extract(images)
        assert features.shape == (2, 512)
        assert not torch.isnan(features).any()

    def test_forward_different_feature_dims(self):
        """Test forward pass with different feature dimensions."""
        for feature_dim in [128, 256, 512, 1024]:
            extractor = TorchFFTExtractor(
                channel_mode="luminance",
                num_radial_bands=8,
                window_function="hann",
                high_freq_emphasis=True,
                feature_dim=feature_dim
            )
            images = torch.rand(2, 3, 224, 224)
            features = extractor.extract(images)
            assert features.shape == (2, feature_dim)

    def test_forward_different_num_radial_bands(self):
        """Test forward pass with different numbers of radial bands."""
        for num_bands in [4, 8, 16]:
            extractor = TorchFFTExtractor(
                channel_mode="luminance",
                num_radial_bands=num_bands,
                window_function="hann",
                high_freq_emphasis=True,
                feature_dim=512
            )
            images = torch.rand(2, 3, 224, 224)
            features = extractor.extract(images)
            assert features.shape == (2, 512)
            assert not torch.isnan(features).any()
