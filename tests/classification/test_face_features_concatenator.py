"""Tests for FaceFeaturesConcatenator."""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from classification import FaceFeaturesConcatenator


class TestFaceFeaturesConcatenatorInitialization:
    """Test FaceFeaturesConcatenator initialization."""

    def test_init_with_default_dims(self):
        """Test initialization with default dimensions."""
        concatenator = FaceFeaturesConcatenator()
        assert concatenator.spatial_feature_dim == 1792
        assert concatenator.frequency_feature_dim == 1024
        assert concatenator.output_dim == 2816

    def test_init_with_custom_dims(self):
        """Test initialization with custom dimensions."""
        concatenator = FaceFeaturesConcatenator(
            spatial_feature_dim=512,
            frequency_feature_dim=256
        )
        assert concatenator.spatial_feature_dim == 512
        assert concatenator.frequency_feature_dim == 256
        assert concatenator.output_dim == 768

    def test_init_invalid_spatial_dim(self):
        """Test initialization with invalid spatial dimension."""
        with pytest.raises(ValueError, match="Spatial feature dimension must be positive"):
            FaceFeaturesConcatenator(spatial_feature_dim=0)

    def test_init_invalid_frequency_dim(self):
        """Test initialization with invalid frequency dimension."""
        with pytest.raises(ValueError, match="Frequency feature dimension must be positive"):
            FaceFeaturesConcatenator(frequency_feature_dim=-10)


class TestFaceFeaturesConcatenatorForward:
    """Test FaceFeaturesConcatenator forward pass."""

    def test_forward_with_4d_spatial_features(
        self,
        batch_spatial_features_4d,
        batch_frequency_features
    ):
        """Test forward pass with 4D spatial features (pooled format)."""
        concatenator = FaceFeaturesConcatenator()
        output = concatenator(batch_spatial_features_4d, batch_frequency_features)

        assert output.shape == (8, 2816)
        assert output.dtype == torch.float32

    def test_forward_with_2d_spatial_features(
        self,
        batch_spatial_features_2d,
        batch_frequency_features
    ):
        """Test forward pass with 2D spatial features (already flattened)."""
        concatenator = FaceFeaturesConcatenator()
        output = concatenator(batch_spatial_features_2d, batch_frequency_features)

        assert output.shape == (8, 2816)
        assert output.dtype == torch.float32

    def test_forward_single_sample(
        self,
        single_sample_spatial_4d,
        single_sample_frequency
    ):
        """Test forward pass with single sample."""
        concatenator = FaceFeaturesConcatenator()
        output = concatenator(single_sample_spatial_4d, single_sample_frequency)

        assert output.shape == (1, 2816)

    def test_forward_concatenation_correctness(self):
        """Test that concatenation is performed correctly along feature dimension."""
        concatenator = FaceFeaturesConcatenator(
            spatial_feature_dim=4,
            frequency_feature_dim=3
        )

        spatial = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # [1, 4]
        frequency = torch.tensor([[5.0, 6.0, 7.0]])     # [1, 3]

        output = concatenator(spatial, frequency)

        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        assert torch.allclose(output, expected)

    def test_forward_preserves_batch_dimension(self):
        """Test that batch dimension is preserved correctly."""
        concatenator = FaceFeaturesConcatenator()

        for batch_size in [1, 4, 16, 32]:
            spatial = torch.randn(batch_size, 1792)
            frequency = torch.randn(batch_size, 1024)
            output = concatenator(spatial, frequency)
            assert output.shape[0] == batch_size

    def test_forward_with_custom_dims(self):
        """Test forward pass with custom feature dimensions."""
        spatial_dim = 512
        frequency_dim = 256
        concatenator = FaceFeaturesConcatenator(
            spatial_feature_dim=spatial_dim,
            frequency_feature_dim=frequency_dim
        )

        spatial = torch.randn(8, spatial_dim)
        frequency = torch.randn(8, frequency_dim)
        output = concatenator(spatial, frequency)

        assert output.shape == (8, spatial_dim + frequency_dim)


class TestFaceFeaturesConcatenatorErrorHandling:
    """Test FaceFeaturesConcatenator error handling."""

    def test_forward_mismatched_batch_sizes(self):
        """Test that mismatched batch sizes raise an error."""
        concatenator = FaceFeaturesConcatenator()

        spatial = torch.randn(8, 1792)
        frequency = torch.randn(4, 1024)  # Different batch size

        with pytest.raises(ValueError, match="Batch sizes must match"):
            concatenator(spatial, frequency)

    def test_forward_wrong_spatial_dimension(self):
        """Test that wrong spatial feature dimension raises an error."""
        concatenator = FaceFeaturesConcatenator(spatial_feature_dim=1792)

        spatial = torch.randn(8, 512)  # Wrong dimension (512 instead of 1792)
        frequency = torch.randn(8, 1024)

        with pytest.raises(ValueError, match="Expected spatial feature dim"):
            concatenator(spatial, frequency)

    def test_forward_wrong_frequency_dimension(self):
        """Test that wrong frequency feature dimension raises an error."""
        concatenator = FaceFeaturesConcatenator(frequency_feature_dim=1024)

        spatial = torch.randn(8, 1792)
        frequency = torch.randn(8, 512)  # Wrong dimension (512 instead of 1024)

        with pytest.raises(ValueError, match="Expected frequency feature dim"):
            concatenator(spatial, frequency)

    def test_forward_invalid_spatial_shape_4d(self):
        """Test that invalid 4D spatial shape raises an error."""
        concatenator = FaceFeaturesConcatenator()

        spatial = torch.randn(8, 1792, 7, 7)  # Not pooled (should be [batch, dim, 1, 1])
        frequency = torch.randn(8, 1024)

        with pytest.raises(ValueError, match="Spatial features must be pooled"):
            concatenator(spatial, frequency)

    def test_forward_invalid_spatial_shape_3d(self):
        """Test that 3D spatial features raise an error."""
        concatenator = FaceFeaturesConcatenator()

        spatial = torch.randn(8, 1792, 1)  # 3D instead of 2D or 4D
        frequency = torch.randn(8, 1024)

        with pytest.raises(ValueError, match="Spatial features must be 2D"):
            concatenator(spatial, frequency)

    def test_forward_invalid_frequency_shape(self):
        """Test that non-2D frequency features raise an error."""
        concatenator = FaceFeaturesConcatenator()

        spatial = torch.randn(8, 1792)
        frequency = torch.randn(8, 1024, 1)  # 3D instead of 2D

        with pytest.raises(ValueError, match="Frequency features must be 2D"):
            concatenator(spatial, frequency)


class TestFaceFeaturesConcatenatorGradientFlow:
    """Test gradient flow through FaceFeaturesConcatenator."""

    def test_gradient_flow(self):
        """Test that gradients flow through concatenation."""
        concatenator = FaceFeaturesConcatenator()

        spatial = torch.randn(8, 1792, requires_grad=True)
        frequency = torch.randn(8, 1024, requires_grad=True)

        output = concatenator(spatial, frequency)
        loss = output.sum()
        loss.backward()

        assert spatial.grad is not None
        assert frequency.grad is not None
        assert spatial.grad.shape == spatial.shape
        assert frequency.grad.shape == frequency.shape

    def test_gradient_magnitude(self):
        """Test that gradients are propagated with correct magnitude."""
        concatenator = FaceFeaturesConcatenator()

        spatial = torch.randn(4, 1792, requires_grad=True)
        frequency = torch.randn(4, 1024, requires_grad=True)

        output = concatenator(spatial, frequency)

        #loss that should produce gradient of 1 for each element
        loss = output.sum()
        loss.backward()

        #gradient should be 1 for all elements (concatenation is identity operation)
        assert torch.allclose(spatial.grad, torch.ones_like(spatial))
        assert torch.allclose(frequency.grad, torch.ones_like(frequency))


class TestFaceFeaturesConcatenatorDeviceCompatibility:
    """Test FaceFeaturesConcatenator device compatibility."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_on_cuda(self):
        """Test forward pass on CUDA device."""
        concatenator = FaceFeaturesConcatenator().cuda()

        spatial = torch.randn(8, 1792).cuda()
        frequency = torch.randn(8, 1024).cuda()

        output = concatenator(spatial, frequency)

        assert output.device.type == 'cuda'
        assert output.shape == (8, 2816)

    def test_forward_on_cpu(self):
        """Test forward pass on CPU device."""
        concatenator = FaceFeaturesConcatenator()

        spatial = torch.randn(8, 1792)
        frequency = torch.randn(8, 1024)

        output = concatenator(spatial, frequency)

        assert output.device.type == 'cpu'
        assert output.shape == (8, 2816)
