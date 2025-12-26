"""
Tests for InferenceEngine module.

This module contains unit tests for the InferenceEngine class and related
utilities for deepfake detection inference.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from inference import InferenceEngine, InferenceResult, VideoInferenceResult


class TestInferenceResultDataclass:
    """Test InferenceResult dataclass."""

    def test_inference_result_creation(self):
        """Test creation of InferenceResult."""
        result = InferenceResult(
            prediction="FAKE",
            confidence=0.87,
            probabilities={"REAL": 0.13, "FAKE": 0.87},
            raw_logits=[-1.5, 2.3]
        )

        assert result.prediction == "FAKE"
        assert result.confidence == 0.87
        assert result.probabilities["FAKE"] == 0.87
        assert len(result.raw_logits) == 2


class TestVideoInferenceResultDataclass:
    """Test VideoInferenceResult dataclass."""

    def test_video_inference_result_creation(self):
        """Test creation of VideoInferenceResult."""
        frame_result = InferenceResult(
            prediction="FAKE",
            confidence=0.87,
            probabilities={"REAL": 0.13, "FAKE": 0.87},
            raw_logits=[-1.5, 2.3]
        )

        result = VideoInferenceResult(
            video_path="test.mp4",
            frame_results=[frame_result],
            aggregate_prediction="FAKE",
            aggregate_confidence=0.87,
            num_frames_analyzed=1
        )

        assert result.video_path == "test.mp4"
        assert result.aggregate_prediction == "FAKE"
        assert result.num_frames_analyzed == 1
        assert len(result.frame_results) == 1


class TestInferenceEngineInitialization:
    """Test InferenceEngine initialization."""

    @patch('inference.inference_engine.DeepFakeDetector')
    @patch('inference.inference_engine.Path.exists')
    def test_init_with_existing_model(self, mock_exists, mock_detector):
        """Test initialization with existing model file."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_detector.return_value = mock_model

        with patch('torch.load') as mock_load:
            mock_load.return_value = {}
            engine = InferenceEngine(
                model_path="test_model.pth",
                auto_download=False
            )

            assert engine.model is not None
            assert isinstance(engine.device, torch.device)

    @patch('inference.inference_engine.DeepFakeDetector')
    @patch('inference.inference_engine.Path.exists')
    def test_init_auto_selects_device(self, mock_exists, mock_detector):
        """Test that device is auto-selected if not provided."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_detector.return_value = mock_model

        with patch('torch.load') as mock_load:
            mock_load.return_value = {}
            engine = InferenceEngine(
                model_path="test_model.pth",
                auto_download=False
            )

            assert engine.device is not None
            assert isinstance(engine.device, torch.device)

    @patch('inference.inference_engine.Path.exists')
    def test_init_raises_error_if_model_not_found(self, mock_exists):
        """Test initialization raises error if model not found and auto_download=False."""
        mock_exists.return_value = False

        with pytest.raises(ValueError, match="Model file not found"):
            InferenceEngine(
                model_path="nonexistent.pth",
                auto_download=False
            )


class TestInferenceEnginePreprocessing:
    """Test InferenceEngine image preprocessing."""

    @patch('inference.inference_engine.DeepFakeDetector')
    @patch('inference.inference_engine.Path.exists')
    def setup_engine(self, mock_exists, mock_detector):
        """Setup a mock engine for testing."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_detector.return_value = mock_model

        with patch('torch.load') as mock_load:
            mock_load.return_value = {}
            return InferenceEngine(
                model_path="test_model.pth",
                auto_download=False,
                use_face_detection=False
            )

    def test_preprocess_image_from_numpy_array(self):
        """Test preprocessing from numpy array."""
        engine = self.setup_engine()

        # Create test image (224, 224, 3)
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        tensor = engine._preprocess_image(image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_image_from_pil(self):
        """Test preprocessing from PIL Image."""
        engine = self.setup_engine()

        # Create test PIL image
        image = Image.new('RGB', (224, 224), color='red')

        tensor = engine._preprocess_image(image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_image_resizes_correctly(self):
        """Test that images are resized to correct dimensions."""
        engine = self.setup_engine()

        # Create larger image
        image = Image.new('RGB', (512, 512), color='blue')

        tensor = engine._preprocess_image(image)

        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_image_raises_error_for_invalid_type(self):
        """Test that preprocessing raises error for invalid input type."""
        engine = self.setup_engine()

        with pytest.raises(ValueError, match="Unsupported image type"):
            engine._preprocess_image(12345)


class TestInferenceEngineInference:
    """Test InferenceEngine inference methods."""

    @patch('inference.inference_engine.DeepFakeDetector')
    @patch('inference.inference_engine.Path.exists')
    def setup_engine(self, mock_exists, mock_detector):
        """Setup a mock engine for testing."""
        mock_exists.return_value = True
        mock_model = MagicMock()

        # Mock forward pass to return logits
        mock_model.return_value = torch.tensor([[1.0, 2.0]])  # Logits for REAL, FAKE

        mock_detector.return_value = mock_model

        with patch('torch.load') as mock_load:
            mock_load.return_value = {}
            return InferenceEngine(
                model_path="test_model.pth",
                auto_download=False,
                use_face_detection=False
            )

    def test_predict_image_returns_inference_result(self):
        """Test that predict_image returns InferenceResult."""
        engine = self.setup_engine()

        # Create test image
        image = Image.new('RGB', (224, 224), color='red')

        result = engine.predict_image(image)

        assert isinstance(result, InferenceResult)
        assert result.prediction in ["REAL", "FAKE"]
        assert 0 <= result.confidence <= 1
        assert "REAL" in result.probabilities
        assert "FAKE" in result.probabilities

    def test_predict_image_output_shape_correct(self):
        """Test that predict_image produces correct prediction."""
        engine = self.setup_engine()

        # Create test image
        image = Image.new('RGB', (224, 224), color='red')

        result = engine.predict_image(image)

        # With logits [1.0, 2.0], FAKE (index 1) should have higher probability
        assert result.prediction == "FAKE"
        assert result.probabilities["FAKE"] > result.probabilities["REAL"]

    def test_predict_batch_returns_list_of_results(self):
        """Test that predict_batch returns list of InferenceResults."""
        engine = self.setup_engine()

        # Create test images
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='blue'),
            Image.new('RGB', (224, 224), color='green')
        ]

        results = engine.predict_batch(images)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, InferenceResult) for r in results)

    def test_predict_batch_processes_all_images(self):
        """Test that predict_batch processes all images."""
        engine = self.setup_engine()

        # Create test images
        images = [Image.new('RGB', (224, 224)) for _ in range(5)]

        results = engine.predict_batch(images)

        assert len(results) == 5


class TestInferenceEngineModelInfo:
    """Test InferenceEngine model information."""

    @patch('inference.inference_engine.DeepFakeDetector')
    @patch('inference.inference_engine.Path.exists')
    def test_get_model_info_returns_dict(self, mock_exists, mock_detector):
        """Test that get_model_info returns dictionary."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.parameters.return_value = [
            torch.randn(10, 10),  # 100 parameters
            torch.randn(5)        # 5 parameters
        ]
        mock_detector.return_value = mock_model

        with patch('torch.load') as mock_load:
            mock_load.return_value = {}
            engine = InferenceEngine(
                model_path="test_model.pth",
                auto_download=False,
                use_face_detection=False
            )

            info = engine.get_model_info()

            assert isinstance(info, dict)
            assert "model_path" in info
            assert "device" in info
            assert "num_parameters" in info
            assert "class_labels" in info

    @patch('inference.inference_engine.DeepFakeDetector')
    @patch('inference.inference_engine.Path.exists')
    def test_get_model_info_has_correct_class_labels(self, mock_exists, mock_detector):
        """Test that get_model_info has correct class labels."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.parameters.return_value = []
        mock_detector.return_value = mock_model

        with patch('torch.load') as mock_load:
            mock_load.return_value = {}
            engine = InferenceEngine(
                model_path="test_model.pth",
                auto_download=False,
                use_face_detection=False
            )

            info = engine.get_model_info()

            assert info["class_labels"] == ["REAL", "FAKE"]


class TestInferenceEngineProcessModelOutput:
    """Test InferenceEngine model output processing."""

    @patch('inference.inference_engine.DeepFakeDetector')
    @patch('inference.inference_engine.Path.exists')
    def setup_engine(self, mock_exists, mock_detector):
        """Setup a mock engine for testing."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_detector.return_value = mock_model

        with patch('torch.load') as mock_load:
            mock_load.return_value = {}
            return InferenceEngine(
                model_path="test_model.pth",
                auto_download=False,
                use_face_detection=False
            )

    def test_process_model_output_with_fake_prediction(self):
        """Test processing output with FAKE prediction."""
        engine = self.setup_engine()

        # Logits favoring FAKE (index 1)
        logits = torch.tensor([[1.0, 3.0]])

        result = engine._process_model_output(logits)

        assert result.prediction == "FAKE"
        assert result.probabilities["FAKE"] > result.probabilities["REAL"]
        assert result.confidence == result.probabilities["FAKE"]

    def test_process_model_output_with_real_prediction(self):
        """Test processing output with REAL prediction."""
        engine = self.setup_engine()

        # Logits favoring REAL (index 0)
        logits = torch.tensor([[3.0, 1.0]])

        result = engine._process_model_output(logits)

        assert result.prediction == "REAL"
        assert result.probabilities["REAL"] > result.probabilities["FAKE"]
        assert result.confidence == result.probabilities["REAL"]

    def test_process_model_output_probabilities_sum_to_one(self):
        """Test that probabilities sum to approximately 1."""
        engine = self.setup_engine()

        logits = torch.tensor([[1.5, 2.5]])

        result = engine._process_model_output(logits)

        prob_sum = sum(result.probabilities.values())
        assert abs(prob_sum - 1.0) < 1e-6
