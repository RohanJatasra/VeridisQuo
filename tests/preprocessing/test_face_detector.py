import pytest
import sys
import os
import numpy as np
import cv2
from unittest.mock import Mock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

# Mock MTCNN before importing face_detector
sys.modules['mtcnn'] = MagicMock()
sys.modules['mtcnn'].MTCNN = MagicMock

import importlib.util

spec = importlib.util.spec_from_file_location(
    "face_detector",
    os.path.join(os.path.dirname(__file__), '../..', 'src', 'preprocessing')
)
from preprocessing import FaceDetector

class TestFaceDetectorInit:
    """Tests for FaceDetector initialization."""

    def test_init_with_custom_parameters(self, mocker):
        """Test initialization with custom parameters."""
        mocker.patch('mtcnn.MTCNN')

        detector = FaceDetector(
            min_face_size=20,
            steps_threshold=[0.5, 0.6, 0.7],
            scale_factor=0.8,
            confidence_threshold=0.85,
            only_keep_top=False
        )

        assert detector.min_face_size == 20
        assert detector.steps_threshold == [0.5, 0.6, 0.7]
        assert detector.scale_factor == 0.8
        assert detector.confidence_threshold == 0.85
        assert detector.only_keep_top is False

class TestFaceDetectorDetectFaces:
    """Tests for FaceDetector.detect_faces() method."""

    def test_detect_single_face(self, mocker):
        """Test detecting a single face returns correct result."""
        detector = FaceDetector(confidence_threshold=0.9)

        detector.detector.detect_faces = mocker.Mock(return_value=[
            {
                'box': [100, 100, 150, 150],
                'confidence': 0.95,
                'keypoints': {'left_eye': (120, 120), 'right_eye': (140, 120)}
            }
        ])

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect_faces(image)

        assert len(results) == 1
        assert results[0]['bbox'] == [100, 100, 150, 150]
        assert results[0]['confidence'] == 0.95
        assert 'keypoints' not in results[0]

    def test_detect_multiple_faces_keep_all(self, mocker):
        """Test detecting multiple faces with only_keep_top=False."""
        detector = FaceDetector(confidence_threshold=0.9, only_keep_top=False)

        detector.detector.detect_faces = mocker.Mock(return_value=[
            {'box': [100, 100, 150, 150], 'confidence': 0.95, 'keypoints': {}},
            {'box': [200, 200, 150, 150], 'confidence': 0.92, 'keypoints': {}},
            {'box': [300, 300, 150, 150], 'confidence': 0.98, 'keypoints': {}}
        ])

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect_faces(image)

        assert len(results) == 3
        # Should be sorted by confidence (descending)
        assert results[0]['confidence'] == 0.98
        assert results[1]['confidence'] == 0.95
        assert results[2]['confidence'] == 0.92

    def test_detect_multiple_faces_keep_top_only(self, mocker):
        """Test detecting multiple faces with only_keep_top=True keeps highest confidence."""
        detector = FaceDetector(confidence_threshold=0.9, only_keep_top=True)

        detector.detector.detect_faces = mocker.Mock(return_value=[
            {'box': [100, 100, 150, 150], 'confidence': 0.95, 'keypoints': {}},
            {'box': [200, 200, 150, 150], 'confidence': 0.92, 'keypoints': {}},
            {'box': [300, 300, 150, 150], 'confidence': 0.98, 'keypoints': {}}
        ])

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect_faces(image)

        assert len(results) == 1
        assert results[0]['confidence'] == 0.98
        assert results[0]['bbox'] == [300, 300, 150, 150]

    def test_detect_no_faces(self, mocker):
        """Test when no faces are detected returns empty list."""
        detector = FaceDetector()

        detector.detector.detect_faces = mocker.Mock(return_value=[])

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect_faces(image)

        assert results == []

    def test_detect_faces_below_confidence_threshold(self, mocker):
        """Test faces below confidence threshold are filtered out."""
        detector = FaceDetector(confidence_threshold=0.9)

        detector.detector.detect_faces = mocker.Mock(return_value=[
            {'box': [100, 100, 150, 150], 'confidence': 0.85, 'keypoints': {}},
            {'box': [200, 200, 150, 150], 'confidence': 0.88, 'keypoints': {}}
        ])

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect_faces(image)

        assert results == []

    def test_detect_faces_with_landmarks(self, mocker):
        """Test detecting faces with return_landmarks=True includes keypoints."""
        detector = FaceDetector()

        detector.detector.detect_faces = mocker.Mock(return_value=[
            {
                'box': [100, 100, 150, 150],
                'confidence': 0.95,
                'keypoints': {
                    'left_eye': (120, 120),
                    'right_eye': (140, 120),
                    'nose': (130, 135),
                    'mouth_left': (125, 150),
                    'mouth_right': (135, 150)
                }
            }
        ])

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect_faces(image, return_landmarks=True)

        assert len(results) == 1
        assert 'keypoints' in results[0]
        assert 'left_eye' in results[0]['keypoints']
        assert 'right_eye' in results[0]['keypoints']

    def test_detect_faces_mixed_confidence(self, mocker):
        """Test detecting faces with mixed confidence levels filters correctly."""
        detector = FaceDetector(confidence_threshold=0.9, only_keep_top=False)

        detector.detector.detect_faces = mocker.Mock(return_value=[
            {'box': [100, 100, 150, 150], 'confidence': 0.85, 'keypoints': {}},
            {'box': [200, 200, 150, 150], 'confidence': 0.95, 'keypoints': {}},
            {'box': [300, 300, 150, 150], 'confidence': 0.88, 'keypoints': {}},
            {'box': [400, 400, 150, 150], 'confidence': 0.92, 'keypoints': {}}
        ])

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect_faces(image)

        assert len(results) == 2
        assert results[0]['confidence'] == 0.95
        assert results[1]['confidence'] == 0.92

    def test_detect_faces_invalid_image_empty(self):
        """Test detect_faces with empty array returns empty list."""
        detector = FaceDetector()
        empty_image = np.array([])

        results = detector.detect_faces(empty_image)
        assert results == []

class TestFaceDetectorDetectSingleFace:
    """Tests for FaceDetector.detect_single_face() method."""

    def test_detect_single_face_found(self, mocker):
        """Test detect_single_face returns highest confidence face."""
        detector = FaceDetector(confidence_threshold=0.9, only_keep_top=False)

        detector.detector.detect_faces = mocker.Mock(return_value=[
            {'box': [100, 100, 150, 150], 'confidence': 0.95, 'keypoints': {}},
            {'box': [200, 200, 150, 150], 'confidence': 0.92, 'keypoints': {}}
        ])

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_single_face(image)

        assert result is not None
        assert result['confidence'] == 0.95
        assert result['bbox'] == [100, 100, 150, 150]

    def test_detect_single_face_not_found(self, mocker):
        """Test detect_single_face returns None when no face detected."""
        detector = FaceDetector()

        detector.detector.detect_faces = mocker.Mock(return_value=[])

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_single_face(image)

        assert result is None

    def test_detect_single_face_with_landmarks(self, mocker):
        """Test detect_single_face with landmarks."""
        detector = FaceDetector()

        detector.detector.detect_faces = mocker.Mock(return_value=[
            {
                'box': [100, 100, 150, 150],
                'confidence': 0.95,
                'keypoints': {'left_eye': (120, 120), 'right_eye': (140, 120)}
            }
        ])

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_single_face(image, return_landmarks=True)

        assert result is not None
        assert 'keypoints' in result
        assert result['keypoints']['left_eye'] == (120, 120)

    def test_detect_single_face_invalid_image(self):
        """Test detect_single_face with invalid image returns None."""
        detector = FaceDetector()

        result = detector.detect_single_face(None)
        assert result is None

class TestFaceDetectorForceRGB:
    """Tests for FaceDetector._force_rgb() method."""

    def test_force_rgb_with_bgr_image(self):
        """Test _force_rgb converts BGR to RGB."""
        detector = FaceDetector()
        # Create BGR image with distinct colors
        bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = 255  # Blue channel

        rgb_image = detector._force_rgb(bgr_image)

        # After conversion, blue should be in the third channel
        assert rgb_image[:, :, 2][0, 0] == 255
        assert rgb_image[:, :, 0][0, 0] == 0

    def test_force_rgb_with_grayscale_image(self):
        """Test _force_rgb with grayscale image returns as-is."""
        detector = FaceDetector()
        gray_image = np.ones((100, 100), dtype=np.uint8) * 128

        result = detector._force_rgb(gray_image)

        # Grayscale image should be returned as-is
        assert result.shape == gray_image.shape
        assert np.array_equal(result, gray_image)

class TestFaceDetectorFormatInputImage:
    """Tests for FaceDetector._format_input_image() method."""

    def test_format_valid_rgb_image(self):
        """Test _format_input_image with valid RGB image."""
        detector = FaceDetector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = detector._format_input_image(image)

        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_format_none_image(self):
        """Test _format_input_image with None raises ValueError."""
        detector = FaceDetector()

        with pytest.raises(ValueError, match="Invalid parameters"):
            detector._format_input_image(None)

    def test_format_empty_image(self):
        """Test _format_input_image with empty array raises ValueError."""
        detector = FaceDetector()
        empty_image = np.array([])

        with pytest.raises(ValueError, match="Invalid parameters"):
            detector._format_input_image(empty_image)

    def test_format_non_numpy_array(self):
        """Test _format_input_image with non-numpy array raises ValueError."""
        detector = FaceDetector()

        with pytest.raises(ValueError, match="Invalid parameters"):
            detector._format_input_image([1, 2, 3])
