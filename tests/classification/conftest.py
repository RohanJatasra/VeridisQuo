"""Pytest fixtures for classification tests."""
import pytest
import torch


@pytest.fixture
def batch_spatial_features_4d():
    """Create batch of spatial features in 4D format (pooled EfficientNet output).

    Returns:
        torch.Tensor: Spatial features [batch, 1792, 1, 1]
    """
    batch_size = 8
    spatial_dim = 1792
    return torch.randn(batch_size, spatial_dim, 1, 1)


@pytest.fixture
def batch_spatial_features_2d():
    """Create batch of spatial features in 2D format (already flattened).

    Returns:
        torch.Tensor: Spatial features [batch, 1792]
    """
    batch_size = 8
    spatial_dim = 1792
    return torch.randn(batch_size, spatial_dim)


@pytest.fixture
def batch_frequency_features():
    """Create batch of frequency features (FrequencyFeatureExtractor output).

    Returns:
        torch.Tensor: Frequency features [batch, 1024]
    """
    batch_size = 8
    frequency_dim = 1024
    return torch.randn(batch_size, frequency_dim)


@pytest.fixture
def batch_concatenated_features():
    """Create batch of concatenated features (for classifier input).

    Returns:
        torch.Tensor: Concatenated features [batch, 2816]
    """
    batch_size = 8
    concatenated_dim = 2816  # 1792 + 1024
    return torch.randn(batch_size, concatenated_dim)


@pytest.fixture
def single_sample_spatial_4d():
    """Create single spatial feature sample in 4D format.

    Returns:
        torch.Tensor: Spatial features [1, 1792, 1, 1]
    """
    spatial_dim = 1792
    return torch.randn(1, spatial_dim, 1, 1)


@pytest.fixture
def single_sample_frequency():
    """Create single frequency feature sample.

    Returns:
        torch.Tensor: Frequency features [1, 1024]
    """
    frequency_dim = 1024
    return torch.randn(1, frequency_dim)


@pytest.fixture
def single_sample_concatenated():
    """Create single concatenated feature sample.

    Returns:
        torch.Tensor: Concatenated features [1, 2816]
    """
    concatenated_dim = 2816
    return torch.randn(1, concatenated_dim)


@pytest.fixture
def labels_binary():
    """Create binary labels for classification (FAKE=0, REAL=1).

    Returns:
        torch.Tensor: Labels [batch] with values 0 or 1
    """
    batch_size = 8
    return torch.randint(0, 2, (batch_size,))
