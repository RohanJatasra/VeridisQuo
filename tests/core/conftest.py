"""Fixtures for core module tests."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
import torch
from utils.config_schemas import ModelArchitectureConfig, SpatialFeatureConfig, FrequencyFeatureConfig, ClassifierConfig


@pytest.fixture
def minimal_model_config():
    """Create minimal ModelArchitectureConfig for testing.
    Returns:
        ModelArchitectureConfig: minimal valid configuration
    """
    spatial_config = SpatialFeatureConfig(
        effnet_pretrained=False,
        path_to_weights=None
    )

    frequency_config = FrequencyFeatureConfig(
        fft_feature_dim=256,
        fft_num_radial_bands=8,
        fft_window_function="hann",
        fft_high_freq_emphasis=True,
        dct_feature_dim=256,
        dct_block_size=8,
        channel_mode="per_channel",
        dct_aggregation_method="frequency_bands",
        dct_num_frequency_bands=10,
        fusion_output_dim=1024,
        fusion_hidden_dims=None,
        fusion_dropout_rate=0.3,
        fusion_use_batch_norm=True,
        path_to_weights=None,
        use_gpu_extractors=False,
        fft_num_azimuthal_sectors=8,
        fft_high_freq_threshold_ratio=0.5,
        fft_artifact_edge_width=5,
        fft_artifact_num_radial_samples=10,
        fft_artifact_center_region_size=10,
        fft_default_smoothing_kernel_size=5,
        fft_energy_band_start_multiplier=2.0,
        fft_energy_band_end_multiplier=4.0,
        fft_epsilon=1e-10,
        dct_epsilon=1e-10
    )

    classifier_config = ClassifierConfig(
        input_dim=2816,
        num_classes=2,
        hidden_dims=None,
        dropout_rate=0.5,
        use_batch_norm=True,
        path_to_weights=None
    )

    config = ModelArchitectureConfig(
        spatial_features=spatial_config,
        frequency_features=frequency_config,
        classifier=classifier_config,
        concat_spatial_features_dim=1792,
        concat_frequency_features_dim=1024,
        score_aggregation_method="mean"
    )

    return config


@pytest.fixture
def sample_model(minimal_model_config):
    """Create DeepFakeDetector instance for testing.
    Returns:
        DeepFakeDetector: model instance with minimal config
    """
    from core.model import DeepFakeDetector
    model = DeepFakeDetector(config=minimal_model_config)
    model.eval()
    return model


@pytest.fixture
def sample_input_batch():
    """Create batch of images [B, 3, 224, 224] for testing.
    Returns:
        torch.Tensor: batch of RGB images in range [0, 1]
    """
    batch_size = 4
    return torch.randn(batch_size, 3, 224, 224)


@pytest.fixture
def sample_input_single():
    """Create single image [1, 3, 224, 224] for testing.
    Returns:
        torch.Tensor: single RGB image
    """
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def normalized_input_batch():
    """Create normalized batch for testing (ImageNet normalization).
    Returns:
        torch.Tensor: normalized batch [B, 3, 224, 224]
    """
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)

    #apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    normalized = (images - mean) / std
    return normalized
