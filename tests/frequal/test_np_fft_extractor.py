"""Tests for NpFFTExtractor."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deepfake-detector'))

from frequal import NpFFTExtractor


class TestNpFFTExtractorInitialization:
    """Test NpFFTExtractor initialization."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        extractor = NpFFTExtractor(
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

    def test_init_different_channel_modes(self):
        """Test initialization with different channel modes."""
        for mode in ["luminance", "grayscale", "per_channel"]:
            extractor = NpFFTExtractor(
                channel_mode=mode,
                num_radial_bands=8,
                window_function="hann",
                high_freq_emphasis=True,
                feature_dim=512
            )
            assert extractor.channel_mode == mode

    def test_init_different_window_functions(self):
        """Test initialization with different window functions."""
        for window in ["hann", "hamming", "blackman", "none", None]:
            extractor = NpFFTExtractor(
                channel_mode="luminance",
                num_radial_bands=8,
                window_function=window,
                high_freq_emphasis=True,
                feature_dim=512
            )
            assert extractor.window_function == window

    def test_init_invalid_channel_mode(self):
        """Test initialization with invalid channel mode."""
        with pytest.raises(ValueError, match="Invalid channel mode"):
            NpFFTExtractor(
                channel_mode="invalid",
                num_radial_bands=8,
                window_function="hann",
                high_freq_emphasis=True,
                feature_dim=512
            )

    def test_init_invalid_num_radial_bands(self):
        """Test initialization with invalid number of radial bands."""
        with pytest.raises(ValueError, match="Number of radial bands must be positive"):
            NpFFTExtractor(
                channel_mode="luminance",
                num_radial_bands=0,
                window_function="hann",
                high_freq_emphasis=True,
                feature_dim=512
            )


class TestNpFFTExtractorFeatureExtraction:
    """Test NpFFTExtractor feature extraction."""

    def test_extract_single_image(self, simple_rgb_image):
        """Test feature extraction from single image."""
        extractor = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        #add batch dimension
        image_batch = simple_rgb_image[np.newaxis, ...]
        features = extractor.extract(image_batch)
        assert features.shape == (1, 512)

    def test_extract_batch(self, batch_rgb_images):
        """Test feature extraction from batch of images."""
        extractor = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        features = extractor.extract(batch_rgb_images)
        assert features.shape == (batch_rgb_images.shape[0], 512)

    def test_extract_output_not_nan(self, simple_rgb_image):
        """Test that extracted features don't contain NaN."""
        extractor = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        image_batch = simple_rgb_image[np.newaxis, ...]
        features = extractor.extract(image_batch)
        assert not np.isnan(features).any()

    def test_extract_output_not_inf(self, simple_rgb_image):
        """Test that extracted features don't contain infinity."""
        extractor = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        image_batch = simple_rgb_image[np.newaxis, ...]
        features = extractor.extract(image_batch)
        assert not np.isinf(features).any()

    def test_extract_deterministic(self, simple_rgb_image):
        """Test that feature extraction is deterministic."""
        extractor = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        image_batch = simple_rgb_image[np.newaxis, ...]
        features1 = extractor.extract(image_batch)
        features2 = extractor.extract(image_batch)
        np.testing.assert_allclose(features1, features2, rtol=1e-5)

    def test_extract_different_images_different_features(self, simple_rgb_image, checkerboard_image):
        """Test that different images produce different features."""
        extractor = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        image1_batch = simple_rgb_image[np.newaxis, ...]
        image2_batch = checkerboard_image[np.newaxis, ...]
        features1 = extractor.extract(image1_batch)
        features2 = extractor.extract(image2_batch)
        #features should be different
        assert not np.allclose(features1, features2, rtol=1e-2)


class TestNpFFTExtractorFrequencyAnalysis:
    """Test FFT frequency analysis properties."""

    def test_checkerboard_high_frequency(self, checkerboard_image, smooth_gradient_image):
        """Test that checkerboard pattern has higher high-frequency energy."""
        extractor = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        checkerboard_batch = checkerboard_image[np.newaxis, ...]
        smooth_batch = smooth_gradient_image[np.newaxis, ...]

        #extract features
        features_checkerboard = extractor.extract(checkerboard_batch)
        features_smooth = extractor.extract(smooth_batch)

        #checkerboard should have different frequency characteristics
        assert not np.allclose(features_checkerboard, features_smooth, rtol=0.1)

    def test_different_channel_modes_produce_different_features(self, simple_rgb_image):
        """Test that different channel modes produce different features."""
        image_batch = simple_rgb_image[np.newaxis, ...]

        extractor_luminance = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        extractor_grayscale = NpFFTExtractor(
            channel_mode="grayscale",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )

        features_luminance = extractor_luminance.extract(image_batch)
        features_grayscale = extractor_grayscale.extract(image_batch)

        #different channel modes should produce different features
        assert not np.allclose(features_luminance, features_grayscale, rtol=1e-2)

    def test_window_function_effect(self, simple_rgb_image):
        """Test that window function affects the features."""
        image_batch = simple_rgb_image[np.newaxis, ...]

        extractor_no_window = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function=None,
            high_freq_emphasis=True,
            feature_dim=512
        )
        extractor_hann = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )

        features_no_window = extractor_no_window.extract(image_batch)
        features_hann = extractor_hann.extract(image_batch)

        #windowing should affect features
        assert not np.allclose(features_no_window, features_hann, rtol=1e-2)


class TestNpFFTExtractorEdgeCases:
    """Test edge cases for NpFFTExtractor."""

    def test_extract_with_small_image(self, small_image):
        """Test feature extraction with small images."""
        extractor = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=4,  #fewer bands for small image
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=256
        )
        image_batch = small_image[np.newaxis, ...]
        features = extractor.extract(image_batch)
        assert features.shape == (1, 256)
        assert not np.isnan(features).any()

    def test_extract_with_large_image(self, large_image):
        """Test feature extraction with large images."""
        extractor = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=16,  #more bands for large image
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=1024
        )
        image_batch = large_image[np.newaxis, ...]
        features = extractor.extract(image_batch)
        assert features.shape == (1, 1024)
        assert not np.isnan(features).any()

    def test_extract_with_different_num_radial_bands(self, simple_rgb_image):
        """Test feature extraction with different numbers of radial bands."""
        image_batch = simple_rgb_image[np.newaxis, ...]

        for num_bands in [4, 8, 16]:
            extractor = NpFFTExtractor(
                channel_mode="luminance",
                num_radial_bands=num_bands,
                window_function="hann",
                high_freq_emphasis=True,
                feature_dim=512
            )
            features = extractor.extract(image_batch)
            assert features.shape == (1, 512)
            assert not np.isnan(features).any()

    def test_extract_with_black_image(self):
        """Test feature extraction with completely black image."""
        extractor = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        black_image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        features = extractor.extract(black_image)
        assert features.shape == (1, 512)
        assert not np.isnan(features).any()

    def test_extract_with_white_image(self):
        """Test feature extraction with completely white image."""
        extractor = NpFFTExtractor(
            channel_mode="luminance",
            num_radial_bands=8,
            window_function="hann",
            high_freq_emphasis=True,
            feature_dim=512
        )
        white_image = np.ones((1, 224, 224, 3), dtype=np.uint8) * 255
        features = extractor.extract(white_image)
        assert features.shape == (1, 512)
        assert not np.isnan(features).any()
