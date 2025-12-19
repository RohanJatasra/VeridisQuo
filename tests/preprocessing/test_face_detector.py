import pytest
import sys
import os
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

# Mock YOLO before importing face_detector
sys.modules['ultralytics'] = MagicMock()
sys.modules['ultralytics'].YOLO = MagicMock

import importlib.util

spec = importlib.util.spec_from_file_location(
    "face_detector",
    os.path.join(os.path.dirname(__file__), '../..', 'src', 'preprocessing')
)
from preprocessing import FaceDetector


def create_mock_yolo_box(x1, y1, x2, y2, confidence):
    """Helper function to create a mock YOLO box object."""
    mock_box = Mock()
    mock_box.xyxy = [Mock()]
    mock_box.xyxy[0].cpu = Mock(return_value=Mock(numpy=Mock(return_value=np.array([x1, y1, x2, y2]))))
    mock_box.conf = [confidence]
    return mock_box


def create_mock_yolo_result(boxes_data):
    """Helper function to create a mock YOLO result object.

    Args:
        boxes_data: List of tuples (x1, y1, x2, y2, confidence)
    """
    mock_result = Mock()

    if boxes_data:
        mock_result.boxes = [create_mock_yolo_box(*box_data) for box_data in boxes_data]
    else:
        mock_result.boxes = []

    return mock_result

=======
>>>>>>> f99f946 (Refactored module export and use)

class TestFaceDetectorInit:
    """Tests for FaceDetector initialization."""

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(face_detector_module, 'YOLO') as mock_yolo_class:
                mock_yolo_instance = Mock()
                mock_yolo_class.return_value = mock_yolo_instance

                detector = FaceDetector(
                    model_path="custom/path/model.pt",
                    min_face_size=20,
                    confidence_threshold=0.85,
                    only_keep_top=False
                )

                assert detector.min_face_size == 20
                assert detector.confidence_threshold == 0.85
                assert detector.only_keep_top is False
                mock_yolo_class.assert_called_once_with("custom/path/model.pt")

    def test_init_model_not_found(self):
        """Test initialization fails when model file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="YOLO model not found"):
                FaceDetector(model_path="nonexistent/model.pt")


class TestFaceDetectorDetectFaces:
    """Tests for FaceDetector.detect_faces() method."""

    def test_detect_single_face(self):
        """Test detecting a single face returns correct result."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(face_detector_module, 'YOLO') as mock_yolo_class:
                mock_yolo_instance = Mock()
                mock_yolo_class.return_value = mock_yolo_instance

                # Mock YOLO detection result
                mock_result = create_mock_yolo_result([(100, 100, 250, 250, 0.95)])
                mock_yolo_instance.return_value = [mock_result]

                detector = FaceDetector(confidence_threshold=0.9)
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                results = detector.detect_faces(image)

                assert len(results) == 1
                assert results[0]['bbox'] == [100, 100, 150, 150]
                assert results[0]['confidence'] == 0.95
                assert 'keypoints' not in results[0]

    def test_detect_multiple_faces_keep_all(self):
        """Test detecting multiple faces with only_keep_top=False."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(face_detector_module, 'YOLO') as mock_yolo_class:
                mock_yolo_instance = Mock()
                mock_yolo_class.return_value = mock_yolo_instance

                mock_result = create_mock_yolo_result([
                    (100, 100, 250, 250, 0.95),
                    (200, 200, 350, 350, 0.92),
                    (300, 300, 450, 450, 0.98)
                ])
                mock_yolo_instance.return_value = [mock_result]

                detector = FaceDetector(confidence_threshold=0.9, only_keep_top=False)
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                results = detector.detect_faces(image)

                assert len(results) == 3
                assert results[0]['confidence'] == 0.98
                assert results[1]['confidence'] == 0.95
                assert results[2]['confidence'] == 0.92

    def test_detect_multiple_faces_keep_top_only(self):
        """Test detecting multiple faces with only_keep_top=True keeps highest confidence."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(face_detector_module, 'YOLO') as mock_yolo_class:
                mock_yolo_instance = Mock()
                mock_yolo_class.return_value = mock_yolo_instance

                mock_result = create_mock_yolo_result([
                    (100, 100, 250, 250, 0.95),
                    (200, 200, 350, 350, 0.92),
                    (300, 300, 450, 450, 0.98)
                ])
                mock_yolo_instance.return_value = [mock_result]

                detector = FaceDetector(confidence_threshold=0.9, only_keep_top=True)
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                results = detector.detect_faces(image)

                assert len(results) == 1
                assert results[0]['confidence'] == 0.98
                assert results[0]['bbox'] == [300, 300, 150, 150]

    def test_detect_no_faces(self):
        """Test when no faces are detected returns empty list."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(face_detector_module, 'YOLO') as mock_yolo_class:
                mock_yolo_instance = Mock()
                mock_yolo_class.return_value = mock_yolo_instance

                mock_result = create_mock_yolo_result([])
                mock_yolo_instance.return_value = [mock_result]

                detector = FaceDetector()
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                results = detector.detect_faces(image)

                assert results == []

    def test_detect_faces_below_confidence_threshold(self):
        """Test faces below confidence threshold are filtered out."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(face_detector_module, 'YOLO') as mock_yolo_class:
                mock_yolo_instance = Mock()
                mock_yolo_class.return_value = mock_yolo_instance

                mock_result = create_mock_yolo_result([
                    (100, 100, 250, 250, 0.85),
                    (200, 200, 350, 350, 0.88)
                ])
                mock_yolo_instance.return_value = [mock_result]

                detector = FaceDetector(confidence_threshold=0.9)
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                results = detector.detect_faces(image)

                assert results == []

    def test_detect_faces_with_landmarks(self):
        """Test detecting faces with return_landmarks=True includes empty keypoints dict."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(face_detector_module, 'YOLO') as mock_yolo_class:
                mock_yolo_instance = Mock()
                mock_yolo_class.return_value = mock_yolo_instance

                mock_result = create_mock_yolo_result([(100, 100, 250, 250, 0.95)])
                mock_yolo_instance.return_value = [mock_result]

                detector = FaceDetector()
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                results = detector.detect_faces(image, return_landmarks=True)

                assert len(results) == 1
                assert 'keypoints' in results[0]
                assert results[0]['keypoints'] == {}

    def test_detect_faces_mixed_confidence(self):
        """Test detecting faces with mixed confidence levels filters correctly."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(face_detector_module, 'YOLO') as mock_yolo_class:
                mock_yolo_instance = Mock()
                mock_yolo_class.return_value = mock_yolo_instance

                mock_result = create_mock_yolo_result([
                    (100, 100, 250, 250, 0.85),
                    (200, 200, 350, 350, 0.95),
                    (300, 300, 450, 450, 0.88),
                    (400, 400, 550, 550, 0.92)
                ])
                mock_yolo_instance.return_value = [mock_result]

                detector = FaceDetector(confidence_threshold=0.9, only_keep_top=False)
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                results = detector.detect_faces(image)

                assert len(results) == 2
                assert results[0]['confidence'] == 0.95
                assert results[1]['confidence'] == 0.92

    def test_detect_faces_invalid_image_empty(self):
        """Test detect_faces with empty array raises ValueError."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(face_detector_module, 'YOLO'):
                detector = FaceDetector()
                empty_image = np.array([])

                with pytest.raises(ValueError, match="Invalid image"):
                    detector.detect_faces(empty_image)

    def test_detect_faces_min_face_size_filter(self):
        """Test that faces smaller than min_face_size are filtered out."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(face_detector_module, 'YOLO') as mock_yolo_class:
                mock_yolo_instance = Mock()
                mock_yolo_class.return_value = mock_yolo_instance

                mock_result = create_mock_yolo_result([
                    (100, 100, 200, 200, 0.95),
                    (300, 300, 330, 330, 0.98)
                ])
                mock_yolo_instance.return_value = [mock_result]

                detector = FaceDetector(min_face_size=40, confidence_threshold=0.7)
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                results = detector.detect_faces(image)

                assert len(results) == 1
                assert results[0]['bbox'] == [100, 100, 100, 100]


class TestFaceDetectorDetectSingleFace:
    """Tests for FaceDetector.detect_single_face() method."""

    def test_detect_single_face_found(self):
        """Test detect_single_face returns highest confidence face."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(face_detector_module, 'YOLO') as mock_yolo_class:
                mock_yolo_instance = Mock()
                mock_yolo_class.return_value = mock_yolo_instance

                mock_result = create_mock_yolo_result([
                    (100, 100, 250, 250, 0.95),
                    (200, 200, 350, 350, 0.92)
                ])
                mock_yolo_instance.return_value = [mock_result]

                detector = FaceDetector(confidence_threshold=0.9, only_keep_top=False)
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                result = detector.detect_single_face(image)

                assert result is not None
                assert result['confidence'] == 0.95
                assert result['bbox'] == [100, 100, 150, 150]

    def test_detect_single_face_not_found(self):
        """Test detect_single_face returns None when no face detected."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(face_detector_module, 'YOLO') as mock_yolo_class:
                mock_yolo_instance = Mock()
                mock_yolo_class.return_value = mock_yolo_instance

                mock_result = create_mock_yolo_result([])
                mock_yolo_instance.return_value = [mock_result]

                detector = FaceDetector()
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                result = detector.detect_single_face(image)

                assert result is None

    def test_detect_single_face_with_landmarks(self):
        """Test detect_single_face with landmarks returns empty keypoints dict."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(face_detector_module, 'YOLO') as mock_yolo_class:
                mock_yolo_instance = Mock()
                mock_yolo_class.return_value = mock_yolo_instance

                mock_result = create_mock_yolo_result([(100, 100, 250, 250, 0.95)])
                mock_yolo_instance.return_value = [mock_result]

                detector = FaceDetector()
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                result = detector.detect_single_face(image, return_landmarks=True)

                assert result is not None
                assert 'keypoints' in result
                assert result['keypoints'] == {}

    def test_detect_single_face_invalid_image(self):
        """Test detect_single_face with invalid image raises ValueError."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(face_detector_module, 'YOLO'):
                detector = FaceDetector()

                with pytest.raises(ValueError, match="Invalid image"):
                    detector.detect_single_face(None)
