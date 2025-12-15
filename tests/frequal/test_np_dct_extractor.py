"""Tests for NpDCTExtractor."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deepfake-detector'))

from frequal import NpDCTExtractor


class TestNpDCTExtractorInitialization:
    """Test NpDCTExtractor initialization."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        extractor = NpDCTExtractor(
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

    def test_init_different_block_sizes(self):
        """Test initialization with different block sizes."""
        for block_size in [8, 16]:
            extractor = NpDCTExtractor(
                block_size=block_size,
                channel_mode="luminance",
                aggregation_method="frequency_bands",
                num_frequency_bands=4,
                feature_dim=512
            )
            assert extractor.block_size == block_size

    def test_init_different_aggregation_methods(self):
        """Test initialization with different aggregation methods."""
        for method in ["frequency_bands", "zigzag", "statistical"]:
            extractor = NpDCTExtractor(
                block_size=8,
                channel_mode="luminance",
                aggregation_method=method,
                num_frequency_bands=4,
                feature_dim=512
            )
            assert extractor.aggregation_method == method

    def test_init_invalid_block_size(self):
        """Test initialization with invalid block size."""
        with pytest.raises(ValueError, match="Block size must be"):
            NpDCTExtractor(
                block_size=32,
                channel_mode="luminance",
                aggregation_method="frequency_bands",
                num_frequency_bands=4,
                feature_dim=512
            )

    def test_init_invalid_aggregation_method(self):
        """Test initialization with invalid aggregation method."""
        with pytest.raises(ValueError, match="Invalid aggregation method"):
            NpDCTExtractor(
                block_size=8,
                channel_mode="luminance",
                aggregation_method="invalid",
                num_frequency_bands=4,
                feature_dim=512
            )


class TestNpDCTExtractorFeatureExtraction:
    """Test NpDCTExtractor feature extraction."""

    def test_extract_single_image(self, simple_rgb_image):
        """Test feature extraction from single image."""
        extractor = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        image_batch = simple_rgb_image[np.newaxis, ...]
        features = extractor.extract(image_batch)
        assert features.shape == (1, 512)

    def test_extract_batch(self, batch_rgb_images):
        """Test feature extraction from batch of images."""
        extractor = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        features = extractor.extract(batch_rgb_images)
        assert features.shape == (batch_rgb_images.shape[0], 512)

    def test_extract_output_not_nan(self, simple_rgb_image):
        """Test that extracted features don't contain NaN."""
        extractor = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        image_batch = simple_rgb_image[np.newaxis, ...]
        features = extractor.extract(image_batch)
        assert not np.isnan(features).any()

    def test_extract_output_not_inf(self, simple_rgb_image):
        """Test that extracted features don't contain infinity."""
        extractor = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        image_batch = simple_rgb_image[np.newaxis, ...]
        features = extractor.extract(image_batch)
        assert not np.isinf(features).any()

    def test_extract_deterministic(self, simple_rgb_image):
        """Test that feature extraction is deterministic."""
        extractor = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        image_batch = simple_rgb_image[np.newaxis, ...]
        features1 = extractor.extract(image_batch)
        features2 = extractor.extract(image_batch)
        np.testing.assert_allclose(features1, features2, rtol=1e-5)


class TestNpDCTExtractorAggregationMethods:
    """Test different aggregation methods."""

    def test_frequency_bands_aggregation(self, simple_rgb_image):
        """Test frequency bands aggregation method."""
        extractor = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        image_batch = simple_rgb_image[np.newaxis, ...]
        features = extractor.extract(image_batch)
        assert features.shape == (1, 512)
        assert not np.isnan(features).any()

    def test_zigzag_aggregation(self, simple_rgb_image):
        """Test zigzag aggregation method."""
        extractor = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="zigzag",
            num_frequency_bands=4,
            feature_dim=512
        )
        image_batch = simple_rgb_image[np.newaxis, ...]
        features = extractor.extract(image_batch)
        assert features.shape == (1, 512)
        assert not np.isnan(features).any()

    def test_statistical_aggregation(self, simple_rgb_image):
        """Test statistical aggregation method."""
        extractor = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="statistical",
            num_frequency_bands=4,
            feature_dim=512
        )
        image_batch = simple_rgb_image[np.newaxis, ...]
        features = extractor.extract(image_batch)
        assert features.shape == (1, 512)
        assert not np.isnan(features).any()

    def test_different_aggregations_produce_different_features(self, simple_rgb_image):
        """Test that different aggregation methods produce different features."""
        image_batch = simple_rgb_image[np.newaxis, ...]

        extractor_freq = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        extractor_zigzag = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="zigzag",
            num_frequency_bands=4,
            feature_dim=512
        )

        features_freq = extractor_freq.extract(image_batch)
        features_zigzag = extractor_zigzag.extract(image_batch)

        #different aggregation methods should produce different raw features
        #(though after projection they might be similar)
        assert not np.allclose(features_freq, features_zigzag, rtol=1e-1)


class TestNpDCTExtractorCompressionArtifacts:
    """Test DCT extractor's ability to detect compression artifacts."""

    def test_detect_jpeg_like_artifacts(self, simple_rgb_image):
        """Test that DCT can detect JPEG-like compression artifacts."""
        extractor = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )

        #original image
        original_batch = simple_rgb_image[np.newaxis, ...]

        #create compressed version by simulating JPEG compression
        import cv2
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, simple_rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
            compressed_image = cv2.imread(temp_path)
            os.unlink(temp_path)

        compressed_batch = compressed_image[np.newaxis, ...]

        features_original = extractor.extract(original_batch)
        features_compressed = extractor.extract(compressed_batch)

        #features should be different due to compression artifacts
        assert not np.allclose(features_original, features_compressed, rtol=0.05)

    def test_smooth_vs_textured_images(self, smooth_gradient_image, checkerboard_image):
        """Test that DCT features differ for smooth vs textured images."""
        extractor = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )

        smooth_batch = smooth_gradient_image[np.newaxis, ...]
        textured_batch = checkerboard_image[np.newaxis, ...]

        features_smooth = extractor.extract(smooth_batch)
        features_textured = extractor.extract(textured_batch)

        #smooth and textured images should have very different DCT features
        assert not np.allclose(features_smooth, features_textured, rtol=0.1)


class TestNpDCTExtractorEdgeCases:
    """Test edge cases for NpDCTExtractor."""

    def test_extract_with_small_image(self, small_image):
        """Test feature extraction with small images."""
        extractor = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=2,
            feature_dim=256
        )
        image_batch = small_image[np.newaxis, ...]
        features = extractor.extract(image_batch)
        assert features.shape == (1, 256)
        assert not np.isnan(features).any()

    def test_extract_with_large_image(self, large_image):
        """Test feature extraction with large images."""
        extractor = NpDCTExtractor(
            block_size=16,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=8,
            feature_dim=1024
        )
        image_batch = large_image[np.newaxis, ...]
        features = extractor.extract(image_batch)
        assert features.shape == (1, 1024)
        assert not np.isnan(features).any()

    def test_extract_with_black_image(self):
        """Test feature extraction with completely black image."""
        extractor = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        black_image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        features = extractor.extract(black_image)
        assert features.shape == (1, 512)
        assert not np.isnan(features).any()

    def test_extract_with_white_image(self):
        """Test feature extraction with completely white image."""
        extractor = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        white_image = np.ones((1, 224, 224, 3), dtype=np.uint8) * 255
        features = extractor.extract(white_image)
        assert features.shape == (1, 512)
        assert not np.isnan(features).any()

    def test_different_block_sizes_produce_different_features(self, simple_rgb_image):
        """Test that different block sizes produce different features."""
        image_batch = simple_rgb_image[np.newaxis, ...]

        extractor_8 = NpDCTExtractor(
            block_size=8,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )
        extractor_16 = NpDCTExtractor(
            block_size=16,
            channel_mode="luminance",
            aggregation_method="frequency_bands",
            num_frequency_bands=4,
            feature_dim=512
        )

        features_8 = extractor_8.extract(image_batch)
        features_16 = extractor_16.extract(image_batch)

        #different block sizes should produce different features
        assert not np.allclose(features_8, features_16, rtol=0.1)
