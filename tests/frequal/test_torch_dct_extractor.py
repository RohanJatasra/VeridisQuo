"""Tests for TorchDCTExtractor."""
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deepfake-detector'))

from frequal import TorchDCTExtractor


class TestTorchDCTExtractorInitialization:
    """Test TorchDCTExtractor initialization."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        assert extractor.block_size == 8
        assert extractor.channel_mode == "luminance"
        assert extractor.aggregation_method == "frequency_bands"
        assert extractor.feature_dim == 512

    def test_init_registers_dct_matrix_buffer(self):
        """Test that DCT matrix buffer is registered."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        assert hasattr(extractor, 'dct_matrix')
        assert isinstance(extractor.dct_matrix, torch.Tensor)
        assert extractor.dct_matrix.shape == (8, 8)

    def test_init_registers_zigzag_indices_buffer(self):
        """Test that zigzag indices buffer is registered."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="zigzag",
            num_frequency_bands=4,
            feature_dim=512
        )
        assert hasattr(extractor, 'zigzag_idx')
        assert isinstance(extractor.zigzag_idx, torch.Tensor)
        assert extractor.zigzag_idx.shape[0] == 64  #8*8

    def test_init_different_block_sizes(self):
        """Test initialization with different block sizes."""
        for block_size in [8, 16]:
            extractor = TorchDCTExtractor(
                block_size=block_size,
                channel_mode="luminance",
                aggregation_method="frequency_bands",
                num_frequency_bands=4,
                feature_dim=512
            )
            assert extractor.dct_matrix.shape == (block_size, block_size)


class TestTorchDCTExtractorForward:
    """Test TorchDCTExtractor forward pass."""

    def test_forward_single_image(self, batch_torch_images):
        """Test forward pass with single image."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        single_image = batch_torch_images[0:1]
        features = extractor.extract(single_image)
        assert features.shape == (1, 512)

    def test_forward_batch(self, batch_torch_images):
        """Test forward pass with batch of images."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        features = extractor.extract(batch_torch_images)
        assert features.shape == (batch_torch_images.shape[0], 512)

    def test_forward_output_not_nan(self, batch_torch_images):
        """Test that forward pass doesn't produce NaN."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        features = extractor.extract(batch_torch_images)
        assert not torch.isnan(features).any()

    def test_forward_output_not_inf(self, batch_torch_images):
        """Test that forward pass doesn't produce infinity."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        features = extractor.extract(batch_torch_images)
        assert not torch.isinf(features).any()

    def test_forward_deterministic_in_eval(self, batch_torch_images):
        """Test that forward pass is deterministic in eval mode."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        extractor.eval()
        features1 = extractor.extract(batch_torch_images)
        features2 = extractor.extract(batch_torch_images)
        assert torch.allclose(features1, features2, rtol=1e-5)

    def test_forward_gradients_flow(self, batch_torch_images):
        """Test that gradients flow through the network."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
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

        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        extractor = extractor.cuda()
        images = batch_torch_images.cuda()
        features = extractor.extract(images)
        assert features.device.type == 'cuda'
        assert features.shape == (batch_torch_images.shape[0], 512)


class TestTorchDCTExtractorAggregationMethods:
    """Test different aggregation methods."""

    def test_frequency_bands_aggregation(self, batch_torch_images):
        """Test frequency bands aggregation method."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        features = extractor.extract(batch_torch_images)
        assert features.shape == (batch_torch_images.shape[0], 512)
        assert not torch.isnan(features).any()

    def test_zigzag_aggregation(self, batch_torch_images):
        """Test zigzag aggregation method."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="zigzag",
            num_frequency_bands=4,
            feature_dim=512
        )
        features = extractor.extract(batch_torch_images)
        assert features.shape == (batch_torch_images.shape[0], 512)
        assert not torch.isnan(features).any()

    def test_statistical_aggregation(self, batch_torch_images):
        """Test statistical aggregation method."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="statistical",
            num_frequency_bands=4,
            feature_dim=512
        )
        features = extractor.extract(batch_torch_images)
        assert features.shape == (batch_torch_images.shape[0], 512)
        assert not torch.isnan(features).any()

    def test_different_aggregations_produce_different_features(self, batch_torch_images):
        """Test that different aggregation methods produce different features."""
        extractor_freq = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        extractor_zigzag = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="zigzag",
            num_frequency_bands=4,
            feature_dim=512
        )

        extractor_freq.eval()
        extractor_zigzag.eval()

        with torch.no_grad():
            features_freq = extractor_freq.extract(batch_torch_images)
            features_zigzag = extractor_zigzag.extract(batch_torch_images)

        #different aggregation methods should produce different features
        assert not torch.allclose(features_freq, features_zigzag, rtol=0.1)


class TestTorchDCTExtractorEdgeCases:
    """Test edge cases for TorchDCTExtractor."""

    def test_forward_with_normalized_input(self):
        """Test forward pass with normalized input [0, 1]."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        images = torch.rand(2, 3, 224, 224)
        features = extractor.extract(images)
        assert features.shape == (2, 512)
        assert not torch.isnan(features).any()

    def test_forward_with_unnormalized_input(self):
        """Test forward pass with unnormalized input [0, 255]."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        images = torch.rand(2, 3, 224, 224) * 255
        features = extractor.extract(images)
        assert features.shape == (2, 512)
        assert not torch.isnan(features).any()

    def test_forward_with_black_images(self):
        """Test forward pass with black images."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        images = torch.zeros(2, 3, 224, 224)
        features = extractor.extract(images)
        assert features.shape == (2, 512)
        assert not torch.isnan(features).any()

    def test_forward_with_white_images(self):
        """Test forward pass with white images."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        images = torch.ones(2, 3, 224, 224)
        features = extractor.extract(images)
        assert features.shape == (2, 512)
        assert not torch.isnan(features).any()

    def test_different_block_sizes_produce_different_features(self):
        """Test that different block sizes produce different features."""
        images = torch.rand(2, 3, 224, 224)

        extractor_8 = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        extractor_16 = TorchDCTExtractor(
            block_size=16,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )

        extractor_8.eval()
        extractor_16.eval()

        with torch.no_grad():
            features_8 = extractor_8.extract(images)
            features_16 = extractor_16.extract(images)

        #different block sizes should produce different features
        assert not torch.allclose(features_8, features_16, rtol=0.1)

    def test_forward_different_feature_dims(self):
        """Test forward pass with different feature dimensions."""
        images = torch.rand(2, 3, 224, 224)

        for feature_dim in [128, 256, 512, 1024]:
            extractor = TorchDCTExtractor(
                block_size=8,
                channel_mode="luminance",
                aggregation_method="frequency_bands",
                num_frequency_bands=4,
                feature_dim=feature_dim
            )
            features = extractor.extract(images)
            assert features.shape == (2, feature_dim)


class TestTorchDCTExtractorDCTProperties:
    """Test DCT mathematical properties."""

    def test_dct_matrix_orthogonality(self):
        """Test that DCT matrix is orthogonal."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        dct_matrix = extractor.dct_matrix
        #D * D^T should be close to identity
        product = torch.matmul(dct_matrix, dct_matrix.t())
        identity = torch.eye(8)
        assert torch.allclose(product, identity, atol=1e-5)

    def test_zigzag_indices_order(self):
        """Test that zigzag indices follow correct order."""
        extractor = TorchDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="zigzag",
            num_frequency_bands=4,
            feature_dim=512
        )
        zigzag_idx = extractor.zigzag_idx
        #first index should be (0, 0) - DC component
        assert zigzag_idx[0, 0] == 0 and zigzag_idx[0, 1] == 0
        #second index should be (0, 1) or (1, 0)
        assert (zigzag_idx[1, 0] == 0 and zigzag_idx[1, 1] == 1) or \
               (zigzag_idx[1, 0] == 1 and zigzag_idx[1, 1] == 0)
