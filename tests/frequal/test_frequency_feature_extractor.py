"""Tests for FrequencyFeatureExtractor."""
import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from frequal import FrequencyFeatureExtractor


class TestFrequencyFeatureExtractorInitialization:
    """Test FrequencyFeatureExtractor initialization."""

    def test_init_with_default_params(self):
        """Test initialization with default parameters."""
        extractor = FrequencyFeatureExtractor()
        assert extractor.fft_feature_dim == 512
        assert extractor.dct_feature_dim == 512
        assert extractor.combined_dim == 1024
        assert extractor.fusion_output_dim == 1024

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        extractor = FrequencyFeatureExtractor(
            fft_feature_dim=256,
            dct_feature_dim=256,
            fusion_output_dim=128
        )
        assert extractor.fft_feature_dim == 256
        assert extractor.dct_feature_dim == 256
        assert extractor.combined_dim == 512
        assert extractor.fusion_output_dim == 128

    def test_init_with_gpu_extractors(self):
        """Test initialization with GPU extractors."""
        extractor = FrequencyFeatureExtractor(use_gpu_extractors=True)
        from frequal import TorchDCTExtractor, TorchFFTExtractor
        assert isinstance(extractor.dct_extractor, TorchDCTExtractor)
        assert isinstance(extractor.fft_extractor, TorchFFTExtractor)

    def test_init_with_cpu_extractors(self):
        """Test initialization with CPU extractors."""
        extractor = FrequencyFeatureExtractor(use_gpu_extractors=False)
        from frequal import NpDCTExtractor, NpFFTExtractor
        assert isinstance(extractor.dct_extractor, NpDCTExtractor)
        assert isinstance(extractor.fft_extractor, NpFFTExtractor)

    def test_init_with_mlp_params(self):
        """Test initialization with custom MLP parameters."""
        extractor = FrequencyFeatureExtractor(
            fusion_hidden_dims=[512, 256, 128],
            fusion_dropout_rate=0.5,
            fusion_use_batch_norm=False
        )
        assert extractor.fusion_mlp.hidden_dims == [512, 256, 128]

    def test_init_invalid_feature_dims(self):
        """Test initialization with invalid feature dimensions."""
        with pytest.raises(ValueError):
            FrequencyFeatureExtractor(fft_feature_dim=0)

        with pytest.raises(ValueError):
            FrequencyFeatureExtractor(dct_feature_dim=-1)


class TestFrequencyFeatureExtractorForward:
    """Test FrequencyFeatureExtractor forward pass."""

    def test_forward_single_image(self):
        """Test forward pass with single image."""
        extractor = FrequencyFeatureExtractor()
        extractor.eval()
        image = torch.rand(1, 3, 224, 224)
        features = extractor.forward(image)
        assert features.shape == (1, extractor.fusion_output_dim)

    def test_forward_batch(self):
        """Test forward pass with batch of images."""
        extractor = FrequencyFeatureExtractor()
        images = torch.rand(8, 3, 224, 224)
        features = extractor.forward(images)
        assert features.shape == (8, extractor.fusion_output_dim)

    def test_forward_output_not_nan(self):
        """Test that forward pass doesn't produce NaN."""
        extractor = FrequencyFeatureExtractor()
        images = torch.rand(4, 3, 224, 224)
        features = extractor.forward(images)
        assert not torch.isnan(features).any()

    def test_forward_output_not_inf(self):
        """Test that forward pass doesn't produce infinity."""
        extractor = FrequencyFeatureExtractor()
        images = torch.rand(4, 3, 224, 224)
        features = extractor.forward(images)
        assert not torch.isinf(features).any()

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic."""
        extractor = FrequencyFeatureExtractor()
        extractor.eval()
        images = torch.rand(4, 3, 224, 224)
        features1 = extractor.forward(images)
        features2 = extractor.forward(images)
        assert torch.allclose(features1, features2, rtol=1e-5)

    def test_forward_with_parallelize(self):
        """Test forward pass with parallelization."""
        extractor = FrequencyFeatureExtractor(use_gpu_extractors=False)
        extractor.eval()
        images = torch.rand(4, 3, 224, 224)
        features_parallel = extractor.forward(images, parallelize=True)
        features_sequential = extractor.forward(images, parallelize=False)
        #results should be the same
        assert torch.allclose(features_parallel, features_sequential, rtol=1e-5)

    def test_forward_gradients_flow(self):
        """Test that gradients flow through the network."""
        extractor = FrequencyFeatureExtractor()
        images = torch.rand(4, 3, 224, 224, requires_grad=True)
        features = extractor.forward(images)
        loss = features.sum()
        loss.backward()
        assert images.grad is not None

    def test_forward_invalid_input_shape(self):
        """Test forward pass with invalid input shape."""
        extractor = FrequencyFeatureExtractor()

        #wrong number of dimensions
        with pytest.raises(ValueError):
            extractor.forward(torch.rand(3, 224, 224))

        #wrong number of channels
        with pytest.raises(ValueError):
            extractor.forward(torch.rand(4, 1, 224, 224))

        #wrong image size
        with pytest.raises(ValueError):
            extractor.forward(torch.rand(4, 3, 128, 128))


class TestFrequencyFeatureExtractorIntegration:
    """Test integration of DCT and FFT extractors."""

    def test_features_combine_dct_and_fft(self):
        """Test that features combine both DCT and FFT information."""
        extractor = FrequencyFeatureExtractor(
            fft_feature_dim=256,
            dct_feature_dim=256,
            fusion_output_dim=512
        )
        images = torch.rand(4, 3, 224, 224)
        features = extractor.forward(images)

        #features should be fusion output
        assert features.shape == (4, 512)

    def test_different_images_produce_different_features(self):
        """Test that different images produce different features."""
        extractor = FrequencyFeatureExtractor()

        #create two very different images
        smooth_images = torch.ones(2, 3, 224, 224) * 0.5
        noisy_images = torch.rand(2, 3, 224, 224)

        features_smooth = extractor.forward(smooth_images)
        features_noisy = extractor.forward(noisy_images)

        #features should be different
        assert not torch.allclose(features_smooth, features_noisy, rtol=0.1)

    def test_similar_images_produce_similar_features(self):
        """Test that similar images produce similar features."""
        extractor = FrequencyFeatureExtractor()
        extractor.eval()

        images = torch.rand(1, 3, 224, 224)
        #add small noise
        images_noisy = images + torch.randn_like(images) * 0.01

        features1 = extractor.forward(images)
        features2 = extractor.forward(images_noisy)

        #features should be similar but not identical
        assert not torch.allclose(features1, features2, rtol=1e-5)
        #cosine similarity should be moderate (frequency features are sensitive to noise)
        cos_sim = torch.nn.functional.cosine_similarity(features1, features2)
        assert cos_sim.item() > 0.5, f"Cosine similarity too low: {cos_sim.item()}"


class TestFrequencyFeatureExtractorWeightManagement:
    """Test weight saving and loading."""

    def test_save_and_load_weights(self):
        """Test saving and loading weights."""
        extractor = FrequencyFeatureExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            weight_path = os.path.join(tmpdir, "weights.pth")

            #save weights
            extractor.save_weights(weight_path)
            assert os.path.exists(weight_path)

            #create new extractor and load weights
            extractor_new = FrequencyFeatureExtractor()
            extractor_new.load_weights(weight_path)

            #set both to eval mode for deterministic behavior
            extractor.eval()
            extractor_new.eval()

            #test that weights are the same
            images = torch.rand(2, 3, 224, 224)
            with torch.no_grad():
                features1 = extractor.forward(images)
                features2 = extractor_new.forward(images)

            assert torch.allclose(features1, features2, rtol=1e-5)

    def test_save_weights_creates_directory(self):
        """Test that save_weights creates parent directory if needed."""
        extractor = FrequencyFeatureExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            weight_path = os.path.join(tmpdir, "subdir", "weights.pth")

            #parent directory doesn't exist yet
            assert not os.path.exists(os.path.dirname(weight_path))

            #save should create directory
            extractor.save_weights(weight_path)
            assert os.path.exists(weight_path)

    def test_load_weights_invalid_path(self):
        """Test loading weights from invalid path."""
        extractor = FrequencyFeatureExtractor()

        with pytest.raises(ValueError, match="does not exist"):
            extractor.load_weights("/nonexistent/path/weights.pth")

    def test_load_weights_backward_compatibility(self):
        """Test loading old weights with fusion_projection key."""
        extractor = FrequencyFeatureExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            weight_path = os.path.join(tmpdir, "old_weights.pth")

            #create old-style weights
            old_state_dict = {
                'fusion_projection': extractor.fusion_mlp.mlp[0].state_dict(),
                'dct_projection': extractor.dct_extractor.projection.state_dict(),
                'fft_projection': extractor.fft_extractor.projection.state_dict(),
            }
            torch.save(old_state_dict, weight_path)

            #should load without error
            extractor.load_weights(weight_path)


class TestFrequencyFeatureExtractorRobustness:
    """Test robustness of FrequencyFeatureExtractor."""

    def test_forward_with_normalized_images(self):
        """Test forward pass with normalized images [0, 1]."""
        extractor = FrequencyFeatureExtractor()
        images = torch.rand(4, 3, 224, 224)
        features = extractor.forward(images)
        assert features.shape == (4, extractor.fusion_output_dim)
        assert not torch.isnan(features).any()

    def test_forward_with_unnormalized_images(self):
        """Test forward pass with unnormalized images [-1, 1]."""
        extractor = FrequencyFeatureExtractor()
        images = torch.rand(4, 3, 224, 224) * 2 - 1
        features = extractor.forward(images)
        assert features.shape == (4, extractor.fusion_output_dim)
        assert not torch.isnan(features).any()

    def test_forward_with_extreme_values(self):
        """Test forward pass with extreme pixel values."""
        extractor = FrequencyFeatureExtractor()

        #all black
        black_images = torch.zeros(2, 3, 224, 224)
        features_black = extractor.forward(black_images)
        assert not torch.isnan(features_black).any()

        #all white
        white_images = torch.ones(2, 3, 224, 224)
        features_white = extractor.forward(white_images)
        assert not torch.isnan(features_white).any()

    def test_forward_gpu_if_available(self):
        """Test forward pass on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        extractor = FrequencyFeatureExtractor(use_gpu_extractors=True)
        extractor = extractor.cuda()
        images = torch.rand(4, 3, 224, 224).cuda()
        features = extractor.forward(images)
        assert features.device.type == 'cuda'
        assert features.shape == (4, extractor.fusion_output_dim)

    def test_forward_different_channel_modes(self):
        """Test forward pass with different channel modes."""
        for channel_mode in ["luminance", "grayscale"]:
            extractor = FrequencyFeatureExtractor(channel_mode=channel_mode)
            images = torch.rand(4, 3, 224, 224)
            features = extractor.forward(images)
            assert features.shape == (4, extractor.fusion_output_dim)
            assert not torch.isnan(features).any()
