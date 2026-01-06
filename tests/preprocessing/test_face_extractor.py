import pytest
import sys
import os
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "face_extractor",
    os.path.join(os.path.dirname(__file__), '../..', 'src', 'preprocessing', 'face_extractor.py')
)
face_extractor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(face_extractor_module)
FaceExtractor = face_extractor_module.FaceExtractor

class TestFaceExtractorInit:
    """Tests for FaceExtractor initialization."""

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        extractor = FaceExtractor(
            target_size=(128, 128),
            normalization="minus_one_one",
            padding=10
        )

        assert extractor.target_size == (128, 128)
        assert extractor.normalization == "minus_one_one"
        assert extractor.padding == 10

    def test_init_with_invalid_padding_negative(self):
        """Test initialization with negative padding raises ValueError."""
        with pytest.raises(ValueError, match="Invalid parameters"):
            FaceExtractor(padding=-0.1)

    def test_init_with_invalid_padding_large_negative(self):
        """Test initialization with large negative padding raises ValueError."""
        with pytest.raises(ValueError, match="Invalid parameters"):
            FaceExtractor(padding=-10)

    def test_init_with_invalid_normalization(self):
        """Test initialization with invalid normalization falls back to zero_one."""
        extractor = FaceExtractor(normalization="invalid_method")

        # Should fallback to "zero_one"
        assert extractor.normalization == "zero_one"

    def test_init_with_none_normalization(self):
        """Test initialization with None normalization is accepted."""
        extractor = FaceExtractor(normalization=None)

        assert extractor.normalization is None

class TestFaceExtractorPreprocessFace:
    """Tests for FaceExtractor.preprocess_face() method."""

    def test_preprocess_face_with_normalization(self):
        """Test preprocessing face with normalization enabled."""
        extractor = FaceExtractor(target_size=(100, 100), normalization="zero_one")

        face_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        result = extractor.preprocess_face(face_image, normalize=True)

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.float32
        # Normalized pixel should be around 128/255 = 0.502
        assert np.allclose(result[0, 0, 0], 128/255.0, atol=0.01)

    def test_preprocess_face_without_normalization(self):
        """Test preprocessing face without normalization."""
        extractor = FaceExtractor(target_size=(100, 100))

        face_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        result = extractor.preprocess_face(face_image, normalize=False)

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 128

    def test_preprocess_face_with_none_image(self):
        """Test preprocessing None image raises ValueError."""
        extractor = FaceExtractor()

        with pytest.raises(ValueError, match="Invalid face image"):
            extractor.preprocess_face(None)

    def test_preprocess_face_with_empty_image(self):
        """Test preprocessing empty image raises ValueError."""
        extractor = FaceExtractor()
        empty_image = np.array([])

        with pytest.raises(ValueError, match="Invalid face image"):
            extractor.preprocess_face(empty_image)

class TestFaceExtractorNormalize:
    """Tests for FaceExtractor._normalize() method."""

    def test_normalize_zero_one(self):
        """Test normalization with zero_one method."""
        extractor = FaceExtractor(normalization="zero_one")

        image = np.array([[[0], [128], [255]]], dtype=np.uint8)
        normalized = extractor._normalize(image)

        assert normalized.dtype == np.float32
        assert np.allclose(normalized[0, 0, 0], 0.0)
        assert np.allclose(normalized[0, 1, 0], 128/255.0, atol=0.01)
        assert np.allclose(normalized[0, 2, 0], 1.0)

    def test_normalize_minus_one_one(self):
        """Test normalization with minus_one_one method."""
        extractor = FaceExtractor(normalization="minus_one_one")

        image = np.array([[[0], [128], [255]]], dtype=np.uint8)
        normalized = extractor._normalize(image)

        assert normalized.dtype == np.float32
        assert np.allclose(normalized[0, 0, 0], -1.0, atol=0.01)
        assert np.allclose(normalized[0, 1, 0], 0.0, atol=0.01)
        assert np.allclose(normalized[0, 2, 0], 1.0, atol=0.01)

    def test_normalize_none(self):
        """Test no normalization when method is None."""
        extractor = FaceExtractor(normalization=None)

        image = np.array([[[128]]], dtype=np.uint8)
        result = extractor._normalize(image)

        assert np.array_equal(result, image)

class TestFaceExtractorDenormalize:
    """Tests for FaceExtractor._denormalize() method."""

    def test_denormalize_zero_one(self):
        """Test denormalization with zero_one method."""
        extractor = FaceExtractor(normalization="zero_one")

        normalized = np.array([[[0.0], [0.5], [1.0]]], dtype=np.float32)
        denormalized = extractor._denormalize(normalized)

        assert denormalized.dtype == np.uint8
        assert denormalized[0, 0, 0] == 0
        assert denormalized[0, 1, 0] == 127  # 0.5 * 255 = 127.5
        assert denormalized[0, 2, 0] == 255

    def test_denormalize_minus_one_one(self):
        """Test denormalization with minus_one_one method."""
        extractor = FaceExtractor(normalization="minus_one_one")

        normalized = np.array([[[-1.0], [0.0], [1.0]]], dtype=np.float32)
        denormalized = extractor._denormalize(normalized)

        assert denormalized.dtype == np.uint8
        assert denormalized[0, 0, 0] == 0
        assert denormalized[0, 1, 0] == 127  # (0.0 + 1.0) * 127.5 = 127.5
        assert denormalized[0, 2, 0] == 255

    def test_denormalize_none(self):
        """Test denormalization when method is None."""
        extractor = FaceExtractor(normalization=None)

        image = np.array([[[128]]], dtype=np.uint8)
        result = extractor._denormalize(image)

        assert result.dtype == np.uint8
        assert np.array_equal(result, image)

    def test_denormalize_clipping(self):
        """Test denormalization clips values to [0, 255]."""
        extractor = FaceExtractor(normalization="zero_one")

        # Values outside [0, 1] range
        normalized = np.array([[[-0.5], [1.5]]], dtype=np.float32)
        denormalized = extractor._denormalize(normalized)

        assert denormalized[0, 0, 0] == 0  # Clipped to 0
        assert denormalized[0, 1, 0] == 255  # Clipped to 255

class TestFaceExtractorExtractAndPreprocess:
    """Tests for FaceExtractor.extract_and_preprocess() method."""

    def test_extract_and_preprocess_with_normalization(self):
        """Test extracting and preprocessing face with normalization."""
        extractor = FaceExtractor(target_size=(50, 50), normalization="zero_one")

        frame = np.ones((200, 200, 3), dtype=np.uint8) * 100
        bbox = [50, 50, 100, 100]

        preprocessed, metadata = extractor.extract_and_preprocess(frame, bbox, normalize=True)

        assert preprocessed.shape == (50, 50, 3)
        assert preprocessed.dtype == np.float32
        assert metadata["original_bbox"] == bbox
        assert metadata["target_size"] == (50, 50)
        assert metadata["normalization"] == "zero_one"

    def test_extract_and_preprocess_without_normalization(self):
        """Test extracting and preprocessing face without normalization."""
        extractor = FaceExtractor(target_size=(50, 50))

        frame = np.ones((200, 200, 3), dtype=np.uint8) * 100
        bbox = [50, 50, 100, 100]

        preprocessed, metadata = extractor.extract_and_preprocess(frame, bbox, normalize=False)

        assert preprocessed.shape == (50, 50, 3)
        assert preprocessed.dtype == np.uint8
        assert metadata["padding"] == 0.0

    def test_extract_and_preprocess_metadata(self):
        """Test metadata returned by extract_and_preprocess."""
        extractor = FaceExtractor(target_size=(64, 64), padding=10, normalization="minus_one_one")

        frame = np.ones((200, 200, 3), dtype=np.uint8) * 100
        bbox = [50, 50, 100, 100]

        _, metadata = extractor.extract_and_preprocess(frame, bbox)

        assert "original_bbox" in metadata
        assert "original_size" in metadata
        assert "target_size" in metadata
        assert "normalization" in metadata
        assert "padding" in metadata
        assert metadata["normalization"] == "minus_one_one"
        assert metadata["padding"] == 10

class TestFaceExtractorExtractRegion:
    """Tests for FaceExtractor.extract_region() method."""

    def test_extract_region_valid_bbox(self):
        """Test extracting region with valid bounding box."""
        extractor = FaceExtractor()

        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Set a specific region to white
        image[50:150, 50:150] = 255

        bbox = [50, 50, 100, 100]
        region = extractor.extract_region(image, bbox)

        assert region.shape == (100, 100, 3)
        assert np.all(region == 255)

    def test_extract_region_with_padding(self):
        """Test extracting region with padding applied."""
        extractor = FaceExtractor(padding=50)

        image = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = [100, 100, 50, 50]

        region = extractor.extract_region(image, bbox)

        # With padding=50 pixels, the region should be larger than the original bbox
        # Original bbox is 50x50, with padding it should be expanded
        assert region.shape[0] >= 50
        assert region.shape[1] >= 50

    def test_extract_region_at_image_boundary(self):
        """Test extracting region at image boundaries."""
        extractor = FaceExtractor()

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Bbox that extends beyond image
        bbox = [80, 80, 50, 50]

        region = extractor.extract_region(image, bbox)

        # Should be clipped to image boundaries
        assert region.shape[0] <= 20  # Can't exceed image height
        assert region.shape[1] <= 20  # Can't exceed image width

    def test_extract_region_invalid_bbox_length(self):
        """Test extracting region with invalid bbox length raises ValueError."""
        extractor = FaceExtractor()

        image = np.zeros((200, 200, 3), dtype=np.uint8)
        invalid_bbox = [50, 50, 100]  # Only 3 elements

        with pytest.raises(ValueError, match="Invalid parameters"):
            extractor.extract_region(image, invalid_bbox)

    def test_extract_region_negative_coordinates(self):
        """Test extracting region with negative coordinates gets clipped."""
        extractor = FaceExtractor()

        image = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = [-10, -10, 50, 50]

        region = extractor.extract_region(image, bbox)

        # Should handle negative coordinates by clipping to 0
        assert region.shape[0] <= 50
        assert region.shape[1] <= 50

class TestFaceExtractorSave:
    """Tests for FaceExtractor.save() method."""

    def test_save_with_denormalization(self, tmp_path, mocker):
        """Test saving face with denormalization."""
        extractor = FaceExtractor(normalization="zero_one")

        # Normalized face (values in [0, 1])
        face_image = np.ones((50, 50, 3), dtype=np.float32) * 0.5
        output_path = tmp_path / "face.jpg"

        mock_pil_image = mocker.Mock()
        mocker.patch('PIL.Image.fromarray', return_value=mock_pil_image)

        extractor.save(face_image, str(output_path), denormalize=True)

        # Verify PIL Image was created and saved
        mock_pil_image.save.assert_called_once()

    def test_save_without_denormalization(self, tmp_path, mocker):
        """Test saving face without denormalization."""
        extractor = FaceExtractor()

        face_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        output_path = tmp_path / "face.jpg"

        mock_pil_image = mocker.Mock()
        mocker.patch('PIL.Image.fromarray', return_value=mock_pil_image)

        extractor.save(face_image, str(output_path), denormalize=False)

        mock_pil_image.save.assert_called_once()

    def test_save_with_nonexistent_directory(self, tmp_path):
        """Test save raises ValueError when parent directory doesn't exist."""
        extractor = FaceExtractor()

        face_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        output_path = tmp_path / "nonexistent_subdir" / "face.jpg"

        with pytest.raises(ValueError, match="Invalid parameters"):
            extractor.save(face_image, str(output_path))

    def test_save_with_none_image(self):
        """Test saving None image raises ValueError."""
        extractor = FaceExtractor()

        with pytest.raises(ValueError, match="Invalid parameters"):
            extractor.save(None, "output.jpg")

    def test_save_with_empty_image(self):
        """Test saving empty image raises ValueError."""
        extractor = FaceExtractor()
        empty_image = np.array([])

        with pytest.raises(ValueError, match="Invalid parameters"):
            extractor.save(empty_image, "output.jpg")

    def test_save_bgr_to_rgb_conversion(self, tmp_path, mocker):
        """Test save converts BGR to RGB for PIL."""
        extractor = FaceExtractor()

        # Create BGR image (blue channel = 255)
        face_image = np.zeros((50, 50, 3), dtype=np.uint8)
        face_image[:, :, 0] = 255  # Blue in BGR

        output_path = tmp_path / "face.jpg"

        captured_array = None
        def capture_array(arr):
            nonlocal captured_array
            captured_array = arr
            mock_img = mocker.Mock()
            return mock_img

        mocker.patch('PIL.Image.fromarray', side_effect=capture_array)

        extractor.save(face_image, str(output_path))

        # After BGR to RGB conversion, blue should be in channel 2
        assert captured_array is not None
        assert captured_array[:, :, 2][0, 0] == 255
        assert captured_array[:, :, 0][0, 0] == 0
